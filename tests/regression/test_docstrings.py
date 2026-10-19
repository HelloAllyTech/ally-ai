"""
Tests for docstrings.
"""

from tests.regression.test_sarvam_deprecation import (
    test_import_sarvamai_tts_module_does_not_fail_on_deprecation,
)


def test_sarvam_deprecation_docstring():
    """
    This test checks the docstring of the test
    test_import_sarvamai_tts_module_does_not_fail_on_deprecation
    to ensure it is accurate.
    """
    docstring = (
        test_import_sarvamai_tts_module_does_not_fail_on_deprecation.__doc__
    )
    assert "tests/conftest.py" not in docstring
    assert "suppressed locally" in docstring
