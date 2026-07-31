"""Unit tests for the analytics agent's deterministic parts: the chart
validator, the CSV rendering of result rows, and the two prompt builders. The
Gemini calls themselves are not exercised here."""

from app.core.analytics_agent.agent import (
    MAX_CELL_CHARS,
    rows_to_csv,
    validate_chart,
)
from app.core.analytics_agent.prompt import (
    AgentTurn,
    build_answer_prompt,
    build_plan_prompt,
)
from app.core.analytics_agent.schemas import (
    AnswerOutput,
    ChartSpec,
    ChartType,
    PlanIntent,
    QueryPlan,
)

COLUMNS = ["bucket", "sessions", "language"]


def _output(**chart_kwargs) -> AnswerOutput:
    return AnswerOutput(
        answer="42 sessions.",
        chart=ChartSpec(**chart_kwargs),
    )


class TestValidateChart:
    def test_keeps_a_chart_whose_columns_are_all_present(self):
        out = validate_chart(
            _output(type=ChartType.LINE, x="bucket", y="sessions"),
            COLUMNS,
            row_count=10,
            truncated=False,
        )
        assert out.chart.type == ChartType.LINE

    def test_keeps_a_grouped_chart_when_the_group_column_is_present(self):
        out = validate_chart(
            _output(
                type=ChartType.STACKED_BAR, x="bucket", y="sessions", group="language"
            ),
            COLUMNS,
            row_count=10,
            truncated=False,
        )
        assert out.chart.type == ChartType.STACKED_BAR

    def test_drops_a_chart_naming_an_absent_x_column(self):
        # Would render as an empty plot rather than an error — the exact silent
        # failure this validator exists to catch.
        out = validate_chart(
            _output(type=ChartType.LINE, x="week", y="sessions"),
            COLUMNS,
            row_count=10,
            truncated=False,
        )
        assert out.chart.type == ChartType.NONE

    def test_drops_a_chart_naming_an_absent_y_column(self):
        out = validate_chart(
            _output(type=ChartType.BAR, x="bucket", y="count"),
            COLUMNS,
            row_count=10,
            truncated=False,
        )
        assert out.chart.type == ChartType.NONE

    def test_drops_a_chart_naming_an_absent_group_column(self):
        out = validate_chart(
            _output(type=ChartType.BAR, x="bucket", y="sessions", group="tenant"),
            COLUMNS,
            row_count=10,
            truncated=False,
        )
        assert out.chart.type == ChartType.NONE

    def test_drops_a_chart_over_too_few_rows(self):
        out = validate_chart(
            _output(type=ChartType.LINE, x="bucket", y="sessions"),
            COLUMNS,
            row_count=2,
            truncated=False,
        )
        assert out.chart.type == ChartType.NONE

    def test_drops_a_chart_over_a_truncated_sample(self):
        # A plot of the first N rows of a larger result reads as the whole
        # population; the table (which says it is truncated) carries it instead.
        out = validate_chart(
            _output(type=ChartType.BAR, x="bucket", y="sessions"),
            COLUMNS,
            row_count=500,
            truncated=True,
        )
        assert out.chart.type == ChartType.NONE

    def test_leaves_an_explicit_no_chart_alone(self):
        out = validate_chart(_output(), COLUMNS, row_count=1, truncated=False)
        assert out.chart.type == ChartType.NONE
        assert out.answer == "42 sessions."


class TestRowsToCsv:
    def test_header_then_one_line_per_row(self):
        csv_text = rows_to_csv(
            ["bucket", "sessions"],
            [
                {"bucket": "2026-07-01", "sessions": 12},
                {"bucket": "2026-07-02", "sessions": 9},
            ],
        )
        assert csv_text.splitlines() == [
            "bucket,sessions",
            "2026-07-01,12",
            "2026-07-02,9",
        ]

    def test_null_renders_as_null_not_empty(self):
        # An unmeasured average must not read to the narrator as a zero or a
        # blank; NULL is the only honest rendering.
        csv_text = rows_to_csv(
            ["bucket", "avg_score"], [{"bucket": "w1", "avg_score": None}]
        )
        assert csv_text.splitlines()[1] == "w1,NULL"

    def test_columns_absent_from_a_row_render_as_null(self):
        csv_text = rows_to_csv(["a", "b"], [{"a": 1}])
        assert csv_text.splitlines()[1] == "1,NULL"

    def test_long_cells_are_clamped(self):
        csv_text = rows_to_csv(["note"], [{"note": "x" * (MAX_CELL_CHARS + 50)}])
        assert len(csv_text.splitlines()[1]) == MAX_CELL_CHARS + 1  # + the ellipsis

    def test_no_columns_yields_empty_string(self):
        assert rows_to_csv([], [{"a": 1}]) == ""


class TestBuildPlanPrompt:
    def test_carries_the_catalogue_question_date_and_limit(self):
        prompt = build_plan_prompt(
            "how many sessions last week?",
            schema_catalog="TABLE scenario_sessions(id, started_at)",
            today="2026-07-30",
            row_limit=250,
        )
        assert "scenario_sessions(id, started_at)" in prompt
        assert "how many sessions last week?" in prompt
        assert "2026-07-30" in prompt
        assert "250" in prompt

    def test_states_the_read_only_single_statement_rule(self):
        prompt = build_plan_prompt("q", schema_catalog="t", today="d", row_limit=10)
        assert "Exactly ONE statement" in prompt
        assert "Read-only" in prompt

    def test_history_is_included_oldest_first_for_follow_ups(self):
        prompt = build_plan_prompt(
            "and by language?",
            schema_catalog="t",
            today="d",
            row_limit=10,
            history=[
                AgentTurn(question="first q", sql="SELECT 1", answer="first a"),
                AgentTurn(question="second q", sql="SELECT 2", answer="second a"),
            ],
        )
        assert prompt.index("first q") < prompt.index("second q")
        assert "SELECT 1" in prompt

    def test_no_history_section_when_there_is_no_history(self):
        prompt = build_plan_prompt("q", schema_catalog="t", today="d", row_limit=10)
        assert "CONVERSATION SO FAR" not in prompt


class TestBuildAnswerPrompt:
    def test_carries_question_sql_columns_and_rows(self):
        prompt = build_answer_prompt(
            "how many sessions?",
            sql="SELECT count(*) AS n FROM scenario_sessions LIMIT 1",
            columns=["n"],
            rows_csv="n\n42",
            row_count=1,
            truncated=False,
        )
        assert "how many sessions?" in prompt
        assert "FROM scenario_sessions" in prompt
        assert "rows returned: 1" in prompt
        assert "n\n42" in prompt

    def test_truncation_is_stated_so_a_total_is_marked_a_lower_bound(self):
        prompt = build_answer_prompt(
            "q", sql="s", columns=["a"], rows_csv="a\n1", row_count=500, truncated=True
        )
        assert "TRUNCATED" in prompt

    def test_empty_result_is_labelled_rather_than_left_blank(self):
        prompt = build_answer_prompt(
            "q", sql="s", columns=[], rows_csv="", row_count=0, truncated=False
        )
        assert "(no rows)" in prompt
        assert "(none)" in prompt


class TestQueryPlanDefaults:
    def test_defaults_to_clarify_so_a_blank_plan_never_reads_as_a_query(self):
        plan = QueryPlan()
        assert plan.intent == PlanIntent.CLARIFY
        assert plan.sql == ""
