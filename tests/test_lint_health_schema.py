"""
Regression test for E501 (line too long) in app/schemas/health.py.

flake8 max-line-length is 88 (see setup.cfg); PR #89 introduced two Field(...)
description strings that exceed it on ReadinessCheckResponse.
"""

from pathlib import Path

MAX_LINE_LENGTH = 88
HEALTH_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "app" / "schemas" / "health.py"
)


def test_health_schema_lines_within_max_length():
    lines = HEALTH_SCHEMA_PATH.read_text().splitlines()

    too_long = [
        (lineno, len(line))
        for lineno, line in enumerate(lines, start=1)
        if len(line) > MAX_LINE_LENGTH
    ]

    assert too_long == [], (
        f"lines exceeding {MAX_LINE_LENGTH} characters in "
        f"{HEALTH_SCHEMA_PATH}: {too_long}"
    )
