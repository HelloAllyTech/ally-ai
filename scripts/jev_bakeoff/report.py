"""Phase 0 report: how often does Jev agree with the pinned Gemini judge?

    python scripts/jev_bakeoff/report.py

Reads `<kind>.results.jsonl` from the data dir, writes `report.md` there, and
prints it. Aggregates only; no transcript text reaches the report, so it can
be shared.

Per label it reports agreement and Cohen's kappa overall, by language and by
actor model. Kappa is the headline, not raw agreement: most turns are clean, so
a judge that always answers "no" scores high agreement and near-zero kappa.

The confidence table is the cascade test: if agreement at high confidence is
much better than at low confidence, Jev can answer the confident share and
escalate the rest to Gemini Pro.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from questions import COHERENCE_LEVELS  # noqa: E402

DATA_DIR = Path(os.environ.get("JEV_BAKEOFF_DATA", Path.home() / ".cache" / "ally-jev-bakeoff"))
USD_PER_M_INPUT = 0.042  # list price; free for us today
HIGH_CONFIDENCE = 0.8
MIN_N = 20  # below this a kappa is noise; shown, but flagged

DRIFT_NOULS = ["in_character", "role_inversion", "offered_solution",
               "introduced_new_information", "resistance_briefed"]
DRIFT_CHOICES = ["topic_label", "counselor_utterance_garbled", "ai_reply_failure_mode"]


def load(kind: str) -> list[dict]:
    p = DATA_DIR / f"{kind}.results.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line] if p.exists() else []


def noul_conf(p: float) -> float:
    return abs(2 * p - 1)


def pairs_for_drift(rows):
    """Yield (label, gold, pred, confidence, row) for every judged drift turn."""
    for r in rows:
        g, a = r["gold"], r["answers"]
        if (c := a.get("coherence")) and g.get("coherence"):
            probs = c.get("probabilities", {})
            if probs:
                level = COHERENCE_LEVELS[int(max(probs, key=probs.get))]
                yield "coherence", g["coherence"], level, c.get("confidence"), r
        for q in DRIFT_CHOICES:
            if (c := a.get(q)) and g.get(q) is not None:
                yield q, g[q], c.get("choice"), c.get("confidence"), r
        for q in DRIFT_NOULS:
            if (c := a.get(q)) and g.get(q) is not None:
                p = c.get("noul")
                yield q, bool(g[q]), p >= 0.5, noul_conf(p), r


def pairs_for_groundedness(rows):
    for r in rows:
        a = r["answers"]
        for claim in r["gold"]["claims"]:
            key = f"{claim['kind']}_{claim['claim_index']}"
            if (c := a.get(f"{key}__verdict")) and claim.get("verdict"):
                yield "verdict", claim["verdict"], c.get("choice"), c.get("confidence"), r
            for q in ("quotes_transcript", "quote_is_accurate"):
                c = a.get(f"{key}__{q}")
                if c and claim.get(q) is not None:
                    p = c.get("noul")
                    yield q, bool(claim[q]), p >= 0.5, noul_conf(p), r


def kappa(gold, pred) -> float | None:
    if len(set(gold) | set(pred)) < 2:
        return None  # one class only: kappa is undefined, not zero
    return cohen_kappa_score([str(x) for x in gold], [str(x) for x in pred])


def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def label_table(pairs, group_key) -> list[str]:
    buckets = defaultdict(list)
    for label, g, p, conf, r in pairs:
        buckets[(label, group_key(r))].append((g, p))
    lines = ["| label | group | n | agreement | kappa |", "|---|---|---|---|---|"]
    for (label, grp), gp in sorted(buckets.items()):
        gold, pred = zip(*gp)
        agree = sum(a == b for a, b in gp) / len(gp)
        flag = " (small n)" if len(gp) < MIN_N else ""
        lines.append(f"| {label} | {grp}{flag} | {len(gp)} | {agree:.2f} | {fmt(kappa(gold, pred))} |")
    return lines


def confidence_table(pairs) -> list[str]:
    by = defaultdict(lambda: {"hi": [], "lo": []})
    for label, g, p, conf, r in pairs:
        if conf is None:
            continue
        by[label]["hi" if conf >= HIGH_CONFIDENCE else "lo"].append(g == p)
    lines = [f"| label | share confident (≥{HIGH_CONFIDENCE}) | agreement when confident | "
             "agreement otherwise |", "|---|---|---|---|"]
    for label, d in sorted(by.items()):
        n = len(d["hi"]) + len(d["lo"])
        share = len(d["hi"]) / n if n else None
        hi = sum(d["hi"]) / len(d["hi"]) if d["hi"] else None
        lo = sum(d["lo"]) / len(d["lo"]) if d["lo"] else None
        lines.append(f"| {label} | {fmt(share)} | {fmt(hi)} | {fmt(lo)} |")
    return lines


def ops_lines(rows) -> list[str]:
    if not rows:
        return []
    lat = sorted(r["latency_s"] for r in rows)
    q = statistics.quantiles(lat, n=100) if len(lat) >= 2 else [lat[0]] * 99
    tokens = sum(r["usage"].get("input_tokens", 0) for r in rows)
    models = sorted({r.get("jev_model") or "?" for r in rows})
    return [
        f"- requests: {len(rows)} · Jev model: {', '.join(models)}",
        f"- latency: p50 {q[49]:.2f}s · p90 {q[89]:.2f}s · p99 {q[98]:.2f}s "
        "(from where this script ran, not the prod region)",
        f"- input tokens: {tokens:,} · {tokens / len(rows):,.0f} per request · "
        f"${tokens / 1e6 * USD_PER_M_INPUT:.4f} at list price",
    ]


def section(title, rows, pairs_fn, meta) -> list[str]:
    if not rows:
        return [f"## {title}", "", "No results yet.", ""]
    pairs = list(pairs_fn(rows))
    sessions = len({r["session"] for r in rows})
    out = [f"## {title}", "",
           f"Gemini reference: `{meta.get('judgeModel')}` / `{meta.get('judgePromptVersion')}` · "
           f"{sessions} sessions (test organizations only)", ""]
    out += ops_lines(rows) + [""]
    out += ["### Agreement by language", ""] + label_table(pairs, lambda r: r["language"]) + [""]
    out += ["### Agreement by actor model", ""] + label_table(pairs, lambda r: r["llm_model"]) + [""]
    out += ["### Confidence (cascade test)", ""] + confidence_table(pairs) + [""]
    return out


def main() -> None:
    lines = ["# Jev bake-off (Phase 0)", "",
             "Untuned Jev questions against the pinned Gemini judge. Kappa is the "
             "headline; agreement alone flatters rare labels.", ""]
    for kind, title, fn in [("drift", "Drift judge (per turn)", pairs_for_drift),
                            ("groundedness", "Feedback groundedness (per claim)",
                             pairs_for_groundedness)]:
        meta_path = DATA_DIR / f"{kind}.json"
        meta = {}
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            meta = {k: m.get(k) for k in ("judgeModel", "judgePromptVersion")}
        lines += section(title, load(kind), fn, meta)
    text = "\n".join(lines)
    (DATA_DIR / "report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
