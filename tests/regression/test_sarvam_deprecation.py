import pytest


def test_import_sarvamai_tts_module_does_not_fail_on_deprecation():
    """
    This test ensures that importing the sarvamai TTS module does not
    cause the test suite to fail due to a DeprecationWarning from websockets.
    The warning is suppressed in tests/conftest.py. This test, when run with
    -W error::DeprecationWarning, will pass if the suppression is effective.
    """
    try:
        from sarvamai.text_to_speech_streaming import socket_client

        assert socket_client is not None  # Use the import to avoid F401
    except Exception as e:
        pytest.fail(f"Importing sarvamai TTS module failed unexpectedly: {e}")
