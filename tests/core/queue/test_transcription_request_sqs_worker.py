"""Tests for app/core/queue/transcription_request_sqs_worker.py"""

import asyncio
from types import SimpleNamespace

import pytest

import app.core.queue.transcription_request_sqs_worker as transcription_request_sqs_worker_mod

# from unittest.mock import MagicMock  # Unused import


class TestTranscriptionRequestSQSWorker:
    """Class-based tests for the SQS worker main() flow."""

    @pytest.fixture
    def transcription_request_sqs_worker(self):
        return transcription_request_sqs_worker_mod

    @pytest.fixture
    def patch_settings(self, transcription_request_sqs_worker, monkeypatch):
        # Provide deterministic queue settings
        # settings.QUEUE is a nested object; override its attrs
        test_queue = SimpleNamespace(
            TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL="https://q/input",
        )
        monkeypatch.setattr(transcription_request_sqs_worker.settings, "QUEUE", test_queue, raising=False)

    @pytest.fixture
    def patch_constants(self, transcription_request_sqs_worker, monkeypatch):
        # Provide deterministic worker constants
        test_consts = SimpleNamespace(
            MAX_MESSAGES=5,
            WAIT_TIME_SECONDS=2,
            VISIBILITY_TIMEOUT=30,
            POLLING_INTERVAL=0,
        )
        monkeypatch.setattr(
            transcription_request_sqs_worker, "SQSWorkerConstants", test_consts, raising=False
        )

    @pytest.fixture
    def fakes(self):
        """Provide simple fake classes to capture constructor args and calls."""

        class FakeQueueClient:
            created = False
            closed = False

            @classmethod
            def create_client(cls):
                cls.created = True

            @classmethod
            def get_client(cls):
                return "queue-client"

            @classmethod
            async def close_client(cls):
                cls.closed = True

        class FakeQueueService:
            def __init__(self, client):
                self.client = client

        class FakeAllyCoreClient:
            @classmethod
            async def create_client(cls):
                cls.created = True

            @classmethod
            def get_client(cls):
                return "ally-core-client"

        class FakeAllyCoreService:
            def __init__(self, client):
                self.client = client

        class FakeDeepgramTranscriptionService:
            def __init__(cls):
                return None

        class FakeSarvamTranscriptionService:
            def __init__(cls):
                return None

        class FakeOpenAITranscriptionService:
            def __init__(cls):
                return None

        class FakeEmbeddingClient:
            @classmethod
            def get_client(cls):
                return "embed-client"

        class FakeTextGenClient:
            @classmethod
            def get_client(cls):
                return "textgen-client"

        class FakeEmbeddingService:
            def __init__(self, client):
                self.client = client

        class FakeTextGenService:
            def __init__(self, client, embedding_service):
                self.client = client
                self.embedding_service = embedding_service

        class FakeTranscriptionRequestHandler:
            def __init__(
                self,
                ally_core_service,
                text_generation_service,
                transcription_service,
             ):
                self.ally_core_service = ally_core_service
                self.text_generation_service = text_generation_service
                self.transcription_service = transcription_service

            async def process_transcription_request(
                self, payload
            ):  # pragma: no cover - not used directly
                return None

        class FakeMessageProcessor:
            def __init__(
                self,
                queue_service,
                handler,
                queue_url,
                max_messages,
                wait_time_seconds,
                visibility_timeout,
                polling_interval,
                delete_after_processing,
            ):
                # capture args for assertions
                self.queue_service = queue_service
                self.handler = handler
                self.queue_url = queue_url
                self.max_messages = max_messages
                self.wait_time_seconds = wait_time_seconds
                self.visibility_timeout = visibility_timeout
                self.polling_interval = polling_interval
                self.delete_after_processing = delete_after_processing
                self._task = None

            async def start(self):
                # create a completed task to emulate background run
                async def noop():
                    return None

                self._task = asyncio.create_task(noop())
                await self._task

        return SimpleNamespace(
            QueueClient=FakeQueueClient,
            QueueService=FakeQueueService,
            EmbeddingClient=FakeEmbeddingClient,
            TextGenClient=FakeTextGenClient,
            EmbeddingService=FakeEmbeddingService,
            TextGenService=FakeTextGenService,
            TranscriptionRequestHandler=FakeTranscriptionRequestHandler,
            MessageProcessor=FakeMessageProcessor,
            AllyCoreClient=FakeAllyCoreClient,
            AllyCoreService=FakeAllyCoreService,
            DeepgramTranscriptionService=FakeDeepgramTranscriptionService,
            OpenAITranscriptionService=FakeOpenAITranscriptionService,
            SarvamTranscriptionService=FakeSarvamTranscriptionService
        )

    @pytest.fixture
    def patch_dependencies(self, transcription_request_sqs_worker, monkeypatch, fakes):
        # Patch all external deps referenced inside transcription_request_sqs_worker
        monkeypatch.setattr(transcription_request_sqs_worker, "AllyCoreClient", fakes.AllyCoreClient)
        monkeypatch.setattr(transcription_request_sqs_worker, "AllyCoreService", fakes.AllyCoreService)
        monkeypatch.setattr(transcription_request_sqs_worker, "SQSQueueClient", fakes.QueueClient)
        monkeypatch.setattr(transcription_request_sqs_worker, "SQSQueueService", fakes.QueueService)
        monkeypatch.setattr(transcription_request_sqs_worker, "OpenAIEmbeddingClient", fakes.EmbeddingClient)
        monkeypatch.setattr(
            transcription_request_sqs_worker, "OpenAITextGenerationClient", fakes.TextGenClient
        )
        monkeypatch.setattr(
            transcription_request_sqs_worker, "OpenAIEmbeddingService", fakes.EmbeddingService
        )
        monkeypatch.setattr(
            transcription_request_sqs_worker, "OpenAITextGenerationService", fakes.TextGenService
        )
        monkeypatch.setattr(
            transcription_request_sqs_worker, "TranscriptionRequestHandler", fakes.TranscriptionRequestHandler
        )
        monkeypatch.setattr(transcription_request_sqs_worker, "MessageProcessor", fakes.MessageProcessor)

        monkeypatch.setattr(
            transcription_request_sqs_worker, "DeepgramTranscriptionService", fakes.DeepgramTranscriptionService
        )

        monkeypatch.setattr(
            transcription_request_sqs_worker, "SarvamTranscriptionService", fakes.SarvamTranscriptionService
        )

        monkeypatch.setattr(
            transcription_request_sqs_worker, "OpenAITranscriptionService", fakes.OpenAITranscriptionService
        )
        # No-op client initializer
        monkeypatch.setattr(transcription_request_sqs_worker, "initialize_openai_clients", lambda: None)

    @pytest.mark.asyncio
    async def test_main_happy_path(
        self, transcription_request_sqs_worker, patch_settings, patch_constants, patch_dependencies, fakes
    ):
        # Act
        await transcription_request_sqs_worker.main()

        # Assert queue client lifecycle
        assert fakes.QueueClient.created is True
        assert fakes.QueueClient.closed is True

    @pytest.mark.asyncio
    async def test_main_handles_keyboard_interrupt(
        self,
        transcription_request_sqs_worker,
        patch_settings,
        patch_constants,
        patch_dependencies,
        fakes,
        monkeypatch,
    ):
        # Make MessageProcessor.start raise KeyboardInterrupt
        class RaisingProcessor(fakes.MessageProcessor):
            async def start(self):  # type: ignore[override]
                raise KeyboardInterrupt()

        monkeypatch.setattr(transcription_request_sqs_worker, "MessageProcessor", RaisingProcessor)

        # Act: should NOT raise
        await transcription_request_sqs_worker.main()

        # Always closes client in finally
        assert fakes.QueueClient.closed is True

    @pytest.mark.asyncio
    async def test_main_propagates_unexpected_error_but_cleans_up(
        self,
        transcription_request_sqs_worker,
        patch_settings,
        patch_constants,
        patch_dependencies,
        fakes,
        monkeypatch,
    ):
        class RaisingProcessor(fakes.MessageProcessor):
            async def start(self):  # type: ignore[override]
                raise RuntimeError("boom")

        monkeypatch.setattr(transcription_request_sqs_worker, "MessageProcessor", RaisingProcessor)

        with pytest.raises(RuntimeError):
            await transcription_request_sqs_worker.main()

        # Cleanup still happens
        assert fakes.QueueClient.closed is True

    @pytest.mark.asyncio
    async def test_main_wires_dependencies_and_arguments(
        self, transcription_request_sqs_worker, patch_settings, patch_constants, fakes, monkeypatch
    ):
        # Wrap fakes to capture constructed instances
        constructed = {}

        class CapturingQueueService(fakes.QueueService):
            def __init__(self, client):
                super().__init__(client)
                constructed["queue_service_client"] = client

        class CapturingTranscriptionRequestHandler(fakes.TranscriptionRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                constructed["handler_args"] = kwargs

        class CapturingMessageProcessor(fakes.MessageProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                constructed["processor_kwargs"] = kwargs

        # Patch with capturing fakes and minimal deps
        monkeypatch.setattr(transcription_request_sqs_worker, "AllyCoreClient", fakes.AllyCoreClient)
        monkeypatch.setattr(transcription_request_sqs_worker, "AllyCoreService", fakes.AllyCoreService)
        monkeypatch.setattr(transcription_request_sqs_worker, "SQSQueueClient", fakes.QueueClient)
        monkeypatch.setattr(transcription_request_sqs_worker, "SQSQueueService", CapturingQueueService)
        monkeypatch.setattr(transcription_request_sqs_worker, "OpenAIEmbeddingClient", fakes.EmbeddingClient)
        monkeypatch.setattr(
            transcription_request_sqs_worker, "OpenAITextGenerationClient", fakes.TextGenClient
        )
        monkeypatch.setattr(
            transcription_request_sqs_worker, "OpenAIEmbeddingService", fakes.EmbeddingService
        )
        monkeypatch.setattr(
            transcription_request_sqs_worker, "OpenAITextGenerationService", fakes.TextGenService
        )
        monkeypatch.setattr(
            transcription_request_sqs_worker, "TranscriptionRequestHandler", CapturingTranscriptionRequestHandler
        )
        monkeypatch.setattr(transcription_request_sqs_worker, "MessageProcessor", CapturingMessageProcessor)
        monkeypatch.setattr(transcription_request_sqs_worker, "initialize_openai_clients", lambda: None)

        # Run
        await transcription_request_sqs_worker.main()

        # Verify queue service built with client from SQSQueueClient.get_client()
        assert constructed["queue_service_client"] == "queue-client"

        # Verify handler wiring uses settings
        handler_args = constructed["handler_args"]

        # Verify message processor params from constants
        proc_kwargs = constructed["processor_kwargs"]
        assert (
            proc_kwargs["queue_url"]
            == transcription_request_sqs_worker.settings.QUEUE.TRANSCRIBE_AND_SUMMARIZE_REQUESTS_QUEUE_URL
        )
        assert proc_kwargs["max_messages"] == transcription_request_sqs_worker.SQSWorkerConstants.MAX_MESSAGES
        assert (
            proc_kwargs["wait_time_seconds"]
            == transcription_request_sqs_worker.SQSWorkerConstants.WAIT_TIME_SECONDS
        )
        assert (
            proc_kwargs["visibility_timeout"]
            == transcription_request_sqs_worker.SQSWorkerConstants.VISIBILITY_TIMEOUT
        )
        assert (
            proc_kwargs["polling_interval"]
            == transcription_request_sqs_worker.SQSWorkerConstants.POLLING_INTERVAL
        )
        assert proc_kwargs["delete_after_processing"] is True
