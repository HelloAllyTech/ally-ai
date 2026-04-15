"""Unit tests for structured summary template path selection."""

from app.core.text_generations.openai_text_generation_service import (
    _structured_summary_template_path,
)


def test_structured_summary_template_dictation():
    assert _structured_summary_template_path("DICTATION") == "summary/dictation_summary"
    assert _structured_summary_template_path("dictation") == "summary/dictation_summary"


def test_structured_summary_template_scribe_or_default():
    assert _structured_summary_template_path(None) == "summary/summary"
    assert _structured_summary_template_path("SCRIBE") == "summary/summary"
    assert _structured_summary_template_path("") == "summary/summary"
