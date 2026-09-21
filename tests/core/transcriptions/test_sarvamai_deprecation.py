import warnings
from unittest.mock import MagicMock

from app.core.transcriptions.services.sarvam_service import SarvamTranscriptionService


def test_sarvamai_does_not_emit_websockets_legacy_deprecation_warning(monkeypatch):
    # This test ensures that importing and initializing SarvamTranscriptionService
    # does not emit the deprecated websockets.legacy warning.
    # We expect no DeprecationWarning matching the pattern.
    # We are checking for the ABSENCE of the warning.

    # Mock the settings as it's a dependency for SarvamTranscriptionService
    monkeypatch.setattr("app.core.config.settings.SARVAM.API_KEY", "dummy_key")

    # Mock the AsyncSarvamAI client to prevent actual network calls
    mock_async_sarvam_ai = MagicMock()
    monkeypatch.setattr("sarvamai.AsyncSarvamAI", mock_async_sarvam_ai)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Capture all warnings

        # Instantiate the service, which imports AsyncSarvamAI
        _ = SarvamTranscriptionService()

        # Assert that no DeprecationWarning related to 'websockets.legacy' was issued
        for warning_message in w:
            assert "websockets.legacy is deprecated" not in str(
                warning_message.message
            )
            assert "websockets.WebSocketClientProtocol is deprecated" not in str(
                warning_message.message
            )
