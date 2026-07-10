"""Word/character error rate for the round-trip intelligibility metric
(PRD §4.5): WER(T, ASR(TTS(T))) where T is the LLM's own output text.

Plain Levenshtein over normalized tokens — no external dependency. Use WER
for word-delimited languages, CER where word segmentation is unreliable
(driven by languages.evalConfig.errorRateUnit).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Literal

Unit = Literal["wer", "cer"]

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Case-fold, strip punctuation/symbols, collapse whitespace,
    NFC-normalize — so the metric measures pronunciation, not formatting.

    Punctuation is removed by Unicode category (P*/S*), NOT by a \\w class:
    ``[^\\w\\s]`` would strip Indic combining vowel signs (category Mn),
    destroying Brahmic-script text.
    """
    text = unicodedata.normalize("NFC", text or "")
    text = text.casefold()
    text = "".join(
        " " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text
    )
    return _WS_RE.sub(" ", text).strip()


def _levenshtein(ref: List[str], hyp: List[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(
                prev[j] + 1,  # deletion
                cur[j - 1] + 1,  # insertion
                prev[j - 1] + (0 if r == h else 1),  # substitution
            )
        prev = cur
    return prev[-1]


def error_rate_pct(reference: str, hypothesis: str, unit: Unit = "wer") -> float:
    """(S+D+I)/N × 100 over words (wer) or characters (cer), capped at 100."""
    ref_n = normalize(reference)
    hyp_n = normalize(hypothesis)
    if unit == "cer":
        ref_tokens = [c for c in ref_n if not c.isspace()]
        hyp_tokens = [c for c in hyp_n if not c.isspace()]
    else:
        ref_tokens = ref_n.split()
        hyp_tokens = hyp_n.split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 100.0
    rate = _levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens) * 100
    return round(min(rate, 100.0), 2)
