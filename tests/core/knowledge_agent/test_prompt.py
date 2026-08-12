"""Tests for knowledge-agent prompt assembly.

`format_passages` numbering is the citation contract: the model returns integers and the
agent maps them back positionally. If numbering ever stopped starting at 1, or stopped
matching the order of the list passed in, every citation would silently point at the
wrong passage — the answer would still read fine.
"""

from app.core.knowledge_agent.prompt import (
    ANSWER_PROMPT_PATH,
    MAX_PASSAGE_CHARS,
    TRANSLATE_PROMPT_PATH,
    build_answer_prompt,
    build_crisis_prompt,
    build_translate_prompt,
    format_history,
    format_passages,
)
from app.prompts.resolver import load_template


def passage(**overrides):
    p = {
        "chunk_id": "11111111-1111-1111-1111-111111111111",
        "document_title": "WHO mhGAP Intervention Guide",
        "text": "Ask directly about intent.",
        "page_from": 44,
        "page_to": 44,
        "section_path": "Depression > Assessment",
    }
    p.update(overrides)
    return p


class TestFormatPassages:
    def test_numbering_is_one_based_and_contiguous(self):
        rendered = format_passages(
            [
                passage(text="first"),
                passage(text="second"),
                passage(text="third"),
            ]
        )

        assert "[1] " in rendered
        assert "[2] " in rendered
        assert "[3] " in rendered
        assert "[0]" not in rendered
        # Order must match the input list exactly — the agent resolves positionally.
        assert (
            rendered.index("first") < rendered.index("second") < rendered.index("third")
        )

    def test_single_page_renders_as_p_and_range_as_pp(self):
        one = format_passages([passage(page_from=44, page_to=44)])
        assert "p. 44" in one
        assert "pp." not in one

        span = format_passages([passage(page_from=44, page_to=46)])
        assert "pp. 44-46" in span

    def test_unpaginated_passage_omits_the_page(self):
        """
        page_from is 0 for formats with no pages; 'p. 0' would be a fabricated citation.
        """
        rendered = format_passages([passage(page_from=0, page_to=0)])
        assert "p. 0" not in rendered
        assert "WHO mhGAP Intervention Guide" in rendered

    def test_untitled_source_is_labelled_not_blank(self):
        rendered = format_passages(
            [passage(document_title="", section_path="", page_from=0, page_to=0)]
        )
        assert "untitled source" in rendered

    def test_overlong_passage_is_truncated(self):
        rendered = format_passages([passage(text="x" * (MAX_PASSAGE_CHARS + 500))])
        assert "…" in rendered
        assert len(rendered) < MAX_PASSAGE_CHARS + 300

    def test_no_passages_is_stated_explicitly(self):
        """An empty string here would leave the model with a dangling header."""
        assert "no passages" in format_passages([]).lower()


class TestFormatHistory:
    def test_empty_history_is_stated_explicitly(self):
        assert "no previous messages" in format_history(None).lower()
        assert "no previous messages" in format_history([]).lower()

    def test_roles_are_labelled(self):
        rendered = format_history(
            [
                {"role": "user", "content": "How do I ask about intent?"},
                {"role": "assistant", "content": "Ask directly."},
            ]
        )
        assert "Worker: How do I ask about intent?" in rendered
        assert "Assistant: Ask directly." in rendered

    def test_keeps_the_most_recent_turns(self):
        """Trimming keeps the TAIL.

        A follow-up depends on what was just said, and an old turn that no longer
        applies can make the model answer the previous question again.
        """
        history = [{"role": "user", "content": f"question {i}"} for i in range(20)]
        rendered = format_history(history)

        assert "question 19" in rendered
        assert "question 0" not in rendered

    def test_blank_turns_are_dropped(self):
        rendered = format_history(
            [
                {"role": "user", "content": "   "},
                {"role": "user", "content": "real question"},
            ]
        )
        assert rendered.count("Worker:") == 1


class TestTemplatesResolve:
    """The template files must actually be found at the paths the agent uses.

    `load_template` returns an empty string for a missing file rather than raising, so a
    wrong internal_path would not fail here — it would surface much later as the agent
    refusing to run, or (worse, before the empty-template guard existed) as an unguided
    free-text answer. These assertions pin the paths.
    """

    def test_answer_template_exists(self):
        assert load_template(ANSWER_PROMPT_PATH).strip()

    def test_crisis_template_exists_and_substitutes_the_message(self):
        # A missing template must not silently become an unguided classification: the
        # agent treats an empty prompt as "could not run" rather than "not a crisis",
        # but that only matters if the template is normally there.
        rendered = build_crisis_prompt("I can't keep doing this")

        assert rendered
        assert "I can't keep doing this" in rendered
        assert "{message}" not in rendered

    def test_translate_template_exists(self):
        assert load_template(TRANSLATE_PROMPT_PATH).strip()

    def test_answer_prompt_substitutes_every_placeholder(self):
        rendered = build_answer_prompt(
            "How do I ask about intent?",
            passages=[passage(text="Ask directly about intent.")],
            history=[{"role": "user", "content": "earlier question"}],
            max_chars=1400,
        )

        assert "How do I ask about intent?" in rendered
        assert "Ask directly about intent." in rendered
        assert "earlier question" in rendered
        assert "1400" in rendered
        # An unsubstituted placeholder means the template and the builder disagree on a
        # variable name — the formatter resolves those to empty silently.
        for placeholder in ("{question}", "{passages}", "{history}", "{max_chars}"):
            assert placeholder not in rendered

    def test_translate_prompt_substitutes_the_question(self):
        rendered = build_translate_prompt("आत्महत्या के बारे में कैसे पूछूं?")

        assert "आत्महत्या के बारे में कैसे पूछूं?" in rendered
        assert "{question}" not in rendered
