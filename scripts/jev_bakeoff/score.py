"""Phase 0 scoring: ask Jev the same questions the Gemini judges answered.

    python scripts/jev_bakeoff/score.py            # both judges
    python scripts/jev_bakeoff/score.py --only drift --limit 20

Input is ONLY sessions from test organizations (internal / demo / QA tenants):
export_query.js refuses anything else, because session content is PHI-adjacent
and there is no data agreement with TypeSafe yet. Refuses to run on an export
that does not carry that marker.

Reads the export from the data dir and writes `<kind>.results.jsonl` next to
it. Resumable: units already in the results file are skipped. The key comes
from JEV_API_KEY (TYPESAFE_API_KEY also accepted) or ally-ai/.env.

Drift is asked PER TURN: brief + the prior `--window` transcript lines (0 = all
prior turns). Jev's accuracy drops as state fills with irrelevant content, but
repetition and context lock-in need earlier turns, so the window is a trade-off
the bake-off measures rather than assumes.
Groundedness is asked PER SESSION: the whole transcript is the evidence, and
all of a session's claims share that one state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from questions import DRIFT_QUESTIONS, groundedness_questions  # noqa: E402

DATA_DIR = Path(
    os.environ.get("JEV_BAKEOFF_DATA", Path.home() / ".cache" / "ally-jev-bakeoff")
)
URL = "https://api.typesafe.ai/v1/systemone"
# Pinned, not jev-latest: a moving alias would make two runs incomparable.
MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")
PERSONA_CHARS = 6000


def api_key() -> str:
    key = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        env = Path(__file__).resolve().parents[2] / ".env"
        for line in env.read_text().splitlines():
            if line.startswith("JEV_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("JEV_API_KEY is not set (env or ally-ai/.env)")
    return key


def speaker_line(t: dict) -> str:
    who = "AI_CLIENT" if t.get("role") == "client" else "COUNSELOR"
    tag = f"[turn {t['turn_index']}] " if t.get("turn_index") is not None else ""
    return f"{tag}{who}: {t.get('text', '')}"


def drift_units(export: dict, window: int):
    for s in export["sessions"]:
        transcript = s["transcript"]
        pos = {
            t["turn_index"]: i
            for i, t in enumerate(transcript)
            if t.get("role") == "client"
        }
        for label in s["labels"]:
            i = pos.get(label["turn_index"])
            if i is None:
                continue
            prev = transcript[:i]
            counselor = next(
                (t["text"] for t in reversed(prev) if t.get("role") != "client"), ""
            )
            state = {
                "language": s["language"],
                "client_brief": (s.get("persona") or "")[:PERSONA_CHARS],
                "recent_conversation": [
                    speaker_line(t) for t in (prev[-window:] if window else prev)
                ],
                "turn_being_judged": {
                    "COUNSELOR": counselor,
                    "AI_CLIENT": transcript[i].get("text", ""),
                },
            }
            yield f"{s['id']}:{label['turn_index']}", s, state, DRIFT_QUESTIONS, label


def groundedness_units(export: dict):
    for s in export["sessions"]:
        if not s["claims"]:
            continue
        questions = {}
        for c in s["claims"]:
            questions.update(groundedness_questions(c))
        state = {
            "language": s["language"],
            "transcript": [speaker_line(t) for t in s["transcript"]],
        }
        yield s["id"], s, state, questions, {"claims": s["claims"]}


async def ask(client: httpx.AsyncClient, state, questions) -> tuple[dict, float]:
    body = {"state": state, "model": MODEL, "questions": questions}
    for attempt in range(6):
        t0 = time.perf_counter()
        r = await client.post(URL, json=body)
        dt = time.perf_counter() - t0
        if r.status_code in (429, 529) or r.status_code >= 500:
            await asyncio.sleep(2**attempt)
            continue
        r.raise_for_status()
        return r.json(), dt
    r.raise_for_status()
    raise RuntimeError(f"gave up after retries: HTTP {r.status_code}")


async def run(
    kind: str, limit: int | None, concurrency: int, window: int, tag: str
) -> None:
    export = json.loads((DATA_DIR / f"{kind}.json").read_text())
    if export.get("scope") != "test_organizations_only":
        raise SystemExit(
            f"{kind}.json is not a test-organization-only export; re-export it"
        )
    out_path = DATA_DIR / f"{kind}{tag}.results.jsonl"
    done = set()
    if out_path.exists():
        done = {
            json.loads(line)["unit"]
            for line in out_path.read_text().splitlines()
            if line
        }

    units = list(
        drift_units(export, window) if kind == "drift" else groundedness_units(export)
    )
    todo = [u for u in units if u[0] not in done][: limit or None]
    print(
        f"{kind}: {len(units)} units, {len(done)} done, running {len(todo)}",
        file=sys.stderr,
    )

    # Well under the published 1,200 req/min even at full concurrency.
    sem = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {api_key()}"}
    failures = 0
    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        with out_path.open("a") as out:

            async def one(unit):
                nonlocal failures
                uid, s, state, questions, gold = unit
                async with sem:
                    try:
                        resp, dt = await ask(client, state, questions)
                    except Exception as e:  # keep going; the report counts gaps
                        failures += 1
                        print(
                            f"  {uid}: {type(e).__name__}: {str(e)[:120]}",
                            file=sys.stderr,
                        )
                        return
                out.write(
                    json.dumps(
                        {
                            "unit": uid,
                            "session": s["id"],
                            "language": s["language"],
                            "llm_model": s["llm_model"],
                            "gold": gold,
                            "answers": resp.get("answers", {}),
                            "usage": resp.get("usage", {}),
                            "jev_model": resp.get("model"),
                            "latency_s": round(dt, 4),
                        }
                    )
                    + "\n"
                )
                out.flush()

            await asyncio.gather(*(one(u) for u in todo))
    print(f"{kind}: finished, {failures} failed", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["drift", "groundedness"])
    ap.add_argument("--limit", type=int, help="max units per judge this run")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--window",
        type=int,
        default=6,
        help="drift: prior transcript lines Jev sees; 0 = all prior turns",
    )
    ap.add_argument(
        "--tag", default="", help="results-file suffix, to keep runs side by side"
    )
    args = ap.parse_args()
    for kind in [args.only] if args.only else ["drift", "groundedness"]:
        if not (DATA_DIR / f"{kind}.json").exists():
            print(
                f"{kind}: no export at {DATA_DIR}; run export_prod.py first",
                file=sys.stderr,
            )
            continue
        asyncio.run(run(kind, args.limit, args.concurrency, args.window, args.tag))


if __name__ == "__main__":
    main()
