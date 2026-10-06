"""Data-driven risk cases (en, hi in Devanagari, Hinglish, mr, ta, kn).

A real model's verdicts cannot be asserted offline, so this does the part that can be:
every case's message and context reach the prompt intact and in the right place, and the
case file itself stays well-formed and keeps covering what the classifier must handle —
indirect phrasing, Indic scripts, romanised Hindi, "my friend is…" and short replies
that only mean something in context. The same file is what a live evaluation run should
iterate over, comparing the verdict to `expect_crisis`, `expect_subject` and `band`.
"""

import json
from pathlib import Path

import pytest

from app.core.helpline.prompt import build_risk_prompt, language_name

CASES_PATH = Path(__file__).parent / "fixtures" / "risk_cases.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]

REQUIRED_KEYS = {
    "id",
    "language",
    "message",
    "recent",
    "expect_crisis",
    "expect_subject",
    "band",
    "note",
}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_prompt_carries_the_message_and_context(case):
    prompt = build_risk_prompt(
        case["message"], recent=case["recent"], language=case["language"]
    )

    start = prompt.index("BEGIN MESSAGE")
    end = prompt.index("END MESSAGE")
    assert case["message"] in prompt[start:end]

    for turn in case["recent"]:
        speaker = "Talker" if turn["role"] == "talker" else "Listener"
        assert f"{speaker}: {turn['content']}" in prompt[:start]

    if not case["recent"]:
        assert "(no earlier messages)" in prompt
    assert f"chose {language_name(case['language'])}" in prompt


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_is_well_formed(case):
    assert REQUIRED_KEYS <= set(case)
    assert case["band"] in {"high", "medium", "low"}
    assert case["expect_subject"] in {"SELF", "OTHER", "UNCLEAR"}
    assert all(t["role"] in {"talker", "listener"} for t in case["recent"])
    # A non-crisis case has no subject, and a crisis case is never a "low" band.
    if not case["expect_crisis"]:
        assert case["expect_subject"] == "UNCLEAR"
        assert case["band"] == "low"
    else:
        assert case["band"] != "low"


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


class TestCoverage:
    """The set must keep covering what the contract says the classifier handles."""

    def test_every_required_language_and_script_is_present(self):
        messages = [c["message"] for c in CASES]

        def has(lo: str, hi: str) -> bool:
            return any(lo <= ch <= hi for m in messages for ch in m)

        assert has("ऀ", "ॿ")  # Devanagari (Hindi, Marathi)
        assert has("஀", "௿")  # Tamil
        assert has("ಀ", "೿")  # Kannada
        assert {"en", "hi", "mr", "ta", "kn"} <= {c["language"] for c in CASES}
        # Hinglish: Hindi chosen, message typed in Latin letters.
        assert any(
            c["language"] == "hi" and c["message"].isascii() and c["expect_crisis"]
            for c in CASES
        )

    def test_someone_elses_risk_is_covered_as_subject_other(self):
        other = [c for c in CASES if c["expect_subject"] == "OTHER"]

        assert len(other) >= 2
        assert all(c["expect_crisis"] for c in other)

    def test_short_replies_that_only_mean_something_in_context_are_covered(self):
        short = [c for c in CASES if len(c["message"].split()) <= 4 and c["recent"]]

        assert any(c["expect_crisis"] for c in short)
        assert any(c["expect_subject"] == "UNCLEAR" for c in short)
        assert all(c["recent"][-1]["role"] == "listener" for c in short)

    def test_indirect_passive_and_false_alarm_cases_are_all_present(self):
        ids = {c["id"] for c in CASES}

        assert {
            "en-indirect-no-point",
            "en-indirect-goodbyes",
            "en-indirect-means",
        } <= ids
        assert "en-passive-ideation" in ids
        assert {"en-idiom", "en-ambiguous-tired"} <= ids
