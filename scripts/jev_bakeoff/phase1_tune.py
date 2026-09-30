"""Phase 1: tune one Jev question's wording against the pinned Gemini judge.

    <venv>/bin/python scripts/jev_bakeoff/phase1_tune.py \\
        --label counselor_utterance_garbled

Uses GEPA's `optimize_anything` (the optimizer behind `dspy.GEPA`) rather than
`dspy.GEPA` itself: DSPy's TypeSafe path puts signature instructions into
Jev's STATE and only ever tunes those, while what needs tuning is the question
and its option descriptions. Jev answers; Gemini 2.5 Pro proposes rewrites.

Needs `dspy`/`gepa` (not project dependencies — run from a separate venv; see
the bake-off notes). Data: the test-organization export in the data dir.

The metric is BALANCED: train and validation hold equal numbers of cases the
judge flagged and cases it called clean, so the mean per-example score is
balanced accuracy, (recall + specificity) / 2. Phase 0 showed why: on raw
agreement, answering "no problem" to everything scores 90%.

Guards on the result, the same two a prompt-registry optimization needs:
  * the option KEYS are the judge's label set and may not change (they play
    the role placeholders play in a prompt template), and
  * no 8-word run from the training texts may appear in the tuned wording, so
    no session text leaks into a question that will be reused everywhere.

Train and validation are split by SESSION, never by turn, so a session's
neighbouring turns can't leak an answer across the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from questions import DRIFT_QUESTIONS, groundedness_questions  # noqa: E402
from score import DATA_DIR, MODEL, URL, api_key, drift_units, speaker_line  # noqa: E402

OUT_DIR = DATA_DIR / "phase1"

# label → (kind, what counts as "the judge flagged a problem", the clean option)
LABELS = {
    "counselor_utterance_garbled": ("drift", lambda v: v != "none", "none"),
    "ai_reply_failure_mode": ("drift", lambda v: v != "none", "none"),
    "verdict": ("groundedness", lambda v: v != "supported", "supported"),
}


def env_file_value(name: str) -> str | None:
    for line in (HERE.parents[1] / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ---------------------------------------------------------------- examples


def drift_examples(label: str) -> list[dict]:
    export = json.loads((DATA_DIR / "drift.json").read_text())
    out = []
    for uid, s, state, _q, gold in drift_units(export, window=0):
        if gold.get(label) is None:
            continue
        out.append(
            {
                "uid": uid,
                "session": s["id"],
                "language": s["language"],
                "state": state,
                "gold": gold[label],
                "evidence": state["turn_being_judged"],
            }
        )
    return out


def groundedness_examples() -> list[dict]:
    export = json.loads((DATA_DIR / "groundedness.json").read_text())
    out = []
    for s in export["sessions"]:
        state = {
            "language": s["language"],
            "transcript": [speaker_line(t) for t in s["transcript"]],
        }
        for c in s["claims"]:
            if not c.get("verdict"):
                continue
            out.append(
                {
                    "uid": f"{s['id']}:{c['kind']}:{c['claim_index']}",
                    "session": s["id"],
                    "language": s["language"],
                    "state": state,
                    "gold": c["verdict"],
                    "claim": c,
                    "evidence": {"claim": c["text"], "kind": c["kind"]},
                }
            )
    return out


def split_balanced(examples, is_problem, per_class: int, seed: int = 0):
    """Session-level split, then equal flagged/clean counts in each half."""

    def half(e):
        return int(hashlib.md5(e["session"].encode()).hexdigest(), 16) % 2

    rng = random.Random(seed)
    splits = []
    for h in (0, 1):
        part = [e for e in examples if half(e) == h]
        pos = [e for e in part if is_problem(e["gold"])]
        neg = [e for e in part if not is_problem(e["gold"])]
        n = min(per_class, len(pos), len(neg))
        splits.append(rng.sample(pos, n) + rng.sample(neg, n))
    return splits  # train, val


# ---------------------------------------------------------------- questions


def seed_candidate(label: str) -> dict:
    if label == "verdict":
        q = next(
            v
            for k, v in groundedness_questions(
                {"kind": "positive", "claim_index": 0, "text": "{claim}"}
            ).items()
            if k.endswith("__verdict")
        )
        instructions = q["instructions"].split("\nCLAIM:")[0]
        instructions = instructions.split(" This is an 'improvement' claim")[0]
    else:
        q = DRIFT_QUESTIONS[label]
        instructions = q["instructions"]
    return {
        "instructions": instructions,
        "criteria": json.dumps(q["criteria"], indent=2, ensure_ascii=False),
    }


def build_question(label: str, cand: dict, ex: dict) -> dict:
    criteria = json.loads(cand["criteria"])
    instructions = cand["instructions"]
    if label == "verdict":
        instructions = (
            f"{instructions}\nCLAIM ({ex['claim']['kind']}): " f"{ex['claim']['text']}"
        )
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


# ---------------------------------------------------------------- Jev


_client = None


def jev(state, question) -> dict:
    global _client
    if _client is None:
        _client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key()}"}, timeout=30
        )
    body = {"state": state, "model": MODEL, "questions": {"q": question}}
    for attempt in range(6):
        r = _client.post(URL, json=body)
        if r.status_code in (429, 529) or r.status_code >= 500:
            time.sleep(2**attempt)
            continue
        if r.status_code == 422:
            return {"error": r.text[:300]}
        r.raise_for_status()
        return r.json()["answers"]["q"]
    return {"error": f"HTTP {r.status_code} after retries"}


# ---------------------------------------------------------------- metric


def make_evaluator(label: str, keys: set[str]):
    _, is_problem, clean = LABELS[label]

    def evaluate(candidate: dict, example: dict):
        try:
            criteria = json.loads(candidate["criteria"])
        except (json.JSONDecodeError, TypeError):
            return 0.0, {
                "error": "criteria is not valid JSON; it must be a JSON object "
                "mapping each option key to its description"
            }
        if not isinstance(criteria, dict) or set(criteria) != keys:
            return 0.0, {
                "error": f"criteria keys must be exactly {sorted(keys)} — they are "
                "the judge's label set and cannot be renamed, added or removed"
            }
        ans = jev(example["state"], build_question(label, candidate, example))
        if "error" in ans:
            return 0.0, {"error": ans["error"]}
        pred = ans.get("choice")
        correct = is_problem(pred) == is_problem(example["gold"])
        info = {
            "reference_label": example["gold"],
            "jev_answer": pred,
            "jev_probabilities": ans.get("probabilities"),
            "language": example["language"],
            "evidence": example["evidence"],
        }
        if not correct:
            info["feedback"] = (
                f"MISSED A PROBLEM: the reference judge said '{example['gold']}', "
                f"Jev said '{pred}'."
                if is_problem(example["gold"])
                else f"FALSE ALARM: the reference judge said '{clean}' (no problem), "
                f"Jev said '{pred}'."
            )
        return (1.0 if correct else 0.0), info

    return evaluate


def scores(label, cand, examples) -> dict:
    _, is_problem, _ = LABELS[label]
    ev = make_evaluator(label, set(json.loads(seed_candidate(label)["criteria"])))
    tp = fn = tn = fp = exact = 0
    for ex in examples:
        s, info = ev(cand, ex)
        pred = info.get("jev_answer")
        if is_problem(ex["gold"]):
            tp += s == 1.0
            fn += s != 1.0
            exact += pred == ex["gold"]
        else:
            tn += s == 1.0
            fp += s != 1.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "recall": round(rec, 3),
        "specificity": round(spec, 3),
        "balanced_accuracy": round((rec + spec) / 2, 3),
        "exact_label_on_flagged": round(exact / (tp + fn), 3) if tp + fn else None,
        "n_flagged": tp + fn,
        "n_clean": tn + fp,
    }


def leaks(cand: dict, examples, n: int = 8) -> list[str]:
    """Any n-word run from a training text that appears in the tuned wording."""
    text = " ".join(
        re.findall(r"\w+", (cand["instructions"] + " " + cand["criteria"]).lower())
    )
    found = []
    for ex in examples:
        src = json.dumps(ex["evidence"], ensure_ascii=False).lower()
        words = re.findall(r"\w+", src)
        for i in range(len(words) - n + 1):
            run = " ".join(words[i : i + n])
            if run in text:
                found.append(run)
                break
    return found


# ---------------------------------------------------------------- main


BACKGROUND = {
    "drift": (
        "Ally runs role-play counselling-training sessions. An AI plays the CLIENT; "
        "the human "
        "COUNSELOR trainee speaks through speech-to-text, so their words may be "
        "garbled. "
        "Sessions are in English, Hindi, Marathi, Kannada and Tamil, often code-mixed; "
        "code-switching and transliteration are NORMAL, not errors. The reference "
        "labels come "
        "from a Gemini 2.5 Pro judge that read the whole conversation. The state Jev "
        "sees "
        "contains client_brief, recent_conversation (all prior turns) and "
        "turn_being_judged."
    ),
    "groundedness": (
        "Ally gives counselling trainees written feedback after a role-play session. "
        "Each "
        "feedback claim is checked against the session transcript. The COUNSELLOR is "
        "the "
        "trainee; the CLIENT is an AI, and nothing the client said is the counsellor's "
        "behaviour. An 'improvement' claim saying the counsellor failed to do "
        "something is "
        "CONTRADICTED if they visibly did it, even once or clumsily. The reference "
        "labels come "
        "from a Gemini 2.5 Pro judge. Transcripts may be in Indian languages or "
        "code-mixed."
    ),
}

OBJECTIVE = (
    "Rewrite this typed question for Jev, a model that answers a multiple-choice "
    "question "
    "about a state with calibrated probabilities, so that its answers match the "
    "reference "
    "judge on BOTH flagged and clean cases. The score is balanced accuracy: missing a "
    "real "
    "problem and raising a false alarm cost the same. `instructions` is the question "
    "text. "
    "`criteria` is a JSON object mapping each option key to its description: keep "
    "EXACTLY the "
    "same keys, and improve only the descriptions (you may use an object per option "
    "with "
    "`what`, `not_for` and `examples` fields). Jev reads questions literally and "
    "struggles "
    "with negations and multi-step conditions, so state each boundary directly. Never "
    "copy "
    "text, names or details from the example sessions: the wording must be general, "
    "because "
    "it will be reused on every session."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=sorted(LABELS), required=True)
    ap.add_argument(
        "--per-class",
        type=int,
        default=60,
        help="flagged (and as many clean) examples in each of train and val",
    )
    ap.add_argument(
        "--budget", type=int, default=1500, help="max Jev calls for the search"
    )
    ap.add_argument("--reflection-lm", default="gemini/gemini-2.5-pro")
    args = ap.parse_args()

    os.environ.setdefault("GEMINI_API_KEY", env_file_value("GEMINI__API_KEY") or "")
    import gepa.optimize_anything as oa
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig

    kind, is_problem, _ = LABELS[args.label]
    examples = (
        drift_examples(args.label) if kind == "drift" else groundedness_examples()
    )
    train, val = split_balanced(examples, is_problem, args.per_class)
    print(
        f"{args.label}: train {len(train)}, val {len(val)} (balanced, split by "
        "session)",
        file=sys.stderr,
    )

    seed = seed_candidate(args.label)
    keys = set(json.loads(seed["criteria"]))
    before = scores(args.label, seed, val)
    print(f"seed on val: {before}", file=sys.stderr)

    run_dir = OUT_DIR / f"{args.label}.run"
    result = oa.optimize_anything(
        seed_candidate=seed,
        evaluator=make_evaluator(args.label, keys),
        dataset=train,
        valset=val,
        objective=OBJECTIVE,
        background=BACKGROUND[kind],
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=args.budget,
                max_workers=8,
                run_dir=str(run_dir),
                cache_evaluation=True,
                raise_on_exception=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm=args.reflection_lm, reflection_minibatch_size=6
            ),
        ),
    )
    best = result.best_candidate
    after = scores(args.label, best, val)
    leaked = leaks(best, train + val)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.label}.json").write_text(
        json.dumps(
            {
                "label": args.label,
                "jev_model": MODEL,
                "reflection_lm": args.reflection_lm,
                "train_size": len(train),
                "val_size": len(val),
                "seed": seed,
                "best": best,
                "val_before": before,
                "val_after": after,
                "leak_check": {"passed": not leaked, "runs_found": len(leaked)},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        json.dumps(
            {
                "label": args.label,
                "val_before": before,
                "val_after": after,
                "leak_check_passed": not leaked,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
