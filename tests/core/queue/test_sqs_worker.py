"""Tests for app/core/queue/sqs_worker.py"""

import asyncio
from types import SimpleNamespace

import pytest

import app.core.queue.sqs_worker as sqs_worker_mod

# from unittest.mock import MagicMock  # Unused import


class TestSQSWorker:
    """Class-based tests for the SQS worker main() flow."""

    @pytest.fixture
    def sqs_worker(self):
        return sqs_worker_mod

    @pytest.fixture
    def patch_settings(self, sqs_worker, monkeypatch):
        # Provide deterministic queue settings
        # settings.QUEUE is a nested object; override its attrs
        test_queue = SimpleNamespace(
            TRANSCRIPTION_RESULTS_QUEUE_URL="https://q/input",
            TRANSCRIBE_AND_SUMMARIZE_RESPONSE_QUEUE_URL="https://q/output",
            TRANSCRIBE_AND_SUMMARIZE_RESULTS_BUCKET="test-bucket",
        )
        monkeypatch.setattr(sqs_worker.settings, "QUEUE", test_queue, raising=False)

    @pytest.fixture
    def patch_constants(self, sqs_worker, monkeypatch):
        # Provide deterministic worker constants
        test_consts = SimpleNamespace(
            MAX_MESSAGES=5,
            WAIT_TIME_SECONDS=2,
            VISIBILITY_TIMEOUT=30,
            POLLING_INTERVAL=0,
        )
        monkeypatch.setattr(
            sqs_worker, "SQSWorkerConstants", test_consts, raising=False
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
            def get_client(cls):
                return "ally-core-client"

        class FakeAllyCoreService:
            def __init__(self, client):
                self.client = client

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

        class FakeS3Service:
            def __init__(self):
                pass

        class FakeTranscriptionHandler:
            def __init__(
                self,
                ally_core_service,
                request_queue_url,
                result_queue_url,
                text_generation_service,
                storage_service,
                bucket_name,
            ):
                self.ally_core_service = ally_core_service
                self.request_queue_url = request_queue_url
                self.result_queue_url = result_queue_url
                self.text_generation_service = text_generation_service
                self.storage_service = storage_service
                self.bucket_name = bucket_name

            async def process_request(
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
            S3Service=FakeS3Service,
            TranscriptionHandler=FakeTranscriptionHandler,
            MessageProcessor=FakeMessageProcessor,
            AllyCoreClient=FakeAllyCoreClient,
            AllyCoreService=FakeAllyCoreService,
        )

    @pytest.fixture
    def patch_dependencies(self, sqs_worker, monkeypatch, fakes):
        # Patch all external deps referenced inside sqs_worker
        monkeypatch.setattr(sqs_worker, "AllyCoreClient", fakes.AllyCoreClient)
        monkeypatch.setattr(sqs_worker, "AllyCoreService", fakes.AllyCoreService)
        monkeypatch.setattr(sqs_worker, "SQSQueueClient", fakes.QueueClient)
        monkeypatch.setattr(sqs_worker, "SQSQueueService", fakes.QueueService)
        monkeypatch.setattr(sqs_worker, "OpenAIEmbeddingClient", fakes.EmbeddingClient)
        monkeypatch.setattr(
            sqs_worker, "OpenAITextGenerationClient", fakes.TextGenClient
        )
        monkeypatch.setattr(
            sqs_worker, "OpenAIEmbeddingService", fakes.EmbeddingService
        )
        monkeypatch.setattr(
            sqs_worker, "OpenAITextGenerationService", fakes.TextGenService
        )
        monkeypatch.setattr(sqs_worker, "S3Service", fakes.S3Service)
        monkeypatch.setattr(
            sqs_worker, "TranscriptionHandler", fakes.TranscriptionHandler
        )
        monkeypatch.setattr(sqs_worker, "MessageProcessor", fakes.MessageProcessor)
        # No-op client initializer
        monkeypatch.setattr(sqs_worker, "initialize_openai_clients", lambda: None)

    @pytest.mark.asyncio
    async def test_main_happy_path(
        self, sqs_worker, patch_settings, patch_constants, patch_dependencies, fakes
    ):
        # Act
        await sqs_worker.main()

        # Assert queue client lifecycle
        assert fakes.QueueClient.created is True
        assert fakes.QueueClient.closed is True

    @pytest.mark.asyncio
    async def test_main_handles_keyboard_interrupt(
        self,
        sqs_worker,
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

        monkeypatch.setattr(sqs_worker, "MessageProcessor", RaisingProcessor)

        # Act: should NOT raise
        await sqs_worker.main()

        # Always closes client in finally
        assert fakes.QueueClient.closed is True

    @pytest.mark.asyncio
    async def test_main_propagates_unexpected_error_but_cleans_up(
        self,
        sqs_worker,
        patch_settings,
        patch_constants,
        patch_dependencies,
        fakes,
        monkeypatch,
    ):
        class RaisingProcessor(fakes.MessageProcessor):
            async def start(self):  # type: ignore[override]
                raise RuntimeError("boom")

        monkeypatch.setattr(sqs_worker, "MessageProcessor", RaisingProcessor)

        with pytest.raises(RuntimeError):
            await sqs_worker.main()

        # Cleanup still happens
        assert fakes.QueueClient.closed is True

    @pytest.mark.asyncio
    async def test_main_wires_dependencies_and_arguments(
        self, sqs_worker, patch_settings, patch_constants, fakes, monkeypatch
    ):
        # Wrap fakes to capture constructed instances
        constructed = {}

        class CapturingQueueService(fakes.QueueService):
            def __init__(self, client):
                super().__init__(client)
                constructed["queue_service_client"] = client

        class CapturingTranscriptionHandler(fakes.TranscriptionHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                constructed["handler_args"] = kwargs

        class CapturingMessageProcessor(fakes.MessageProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                constructed["processor_kwargs"] = kwargs

        # Patch with capturing fakes and minimal deps
        monkeypatch.setattr(sqs_worker, "AllyCoreClient", fakes.AllyCoreClient)
        monkeypatch.setattr(sqs_worker, "AllyCoreService", fakes.AllyCoreService)
        monkeypatch.setattr(sqs_worker, "SQSQueueClient", fakes.QueueClient)
        monkeypatch.setattr(sqs_worker, "SQSQueueService", CapturingQueueService)
        monkeypatch.setattr(sqs_worker, "OpenAIEmbeddingClient", fakes.EmbeddingClient)
        monkeypatch.setattr(
            sqs_worker, "OpenAITextGenerationClient", fakes.TextGenClient
        )
        monkeypatch.setattr(
            sqs_worker, "OpenAIEmbeddingService", fakes.EmbeddingService
        )
        monkeypatch.setattr(
            sqs_worker, "OpenAITextGenerationService", fakes.TextGenService
        )
        monkeypatch.setattr(sqs_worker, "S3Service", fakes.S3Service)
        monkeypatch.setattr(
            sqs_worker, "TranscriptionHandler", CapturingTranscriptionHandler
        )
        monkeypatch.setattr(sqs_worker, "MessageProcessor", CapturingMessageProcessor)
        monkeypatch.setattr(sqs_worker, "initialize_openai_clients", lambda: None)

        # Run
        await sqs_worker.main()

        # Verify queue service built with client from SQSQueueClient.get_client()
        assert constructed["queue_service_client"] == "queue-client"

        # Verify handler wiring uses settings
        handler_args = constructed["handler_args"]
        assert (
            handler_args["request_queue_url"]
            == sqs_worker.settings.QUEUE.TRANSCRIPTION_RESULTS_QUEUE_URL
        )
        assert (
            handler_args["result_queue_url"]
            == sqs_worker.settings.QUEUE.TRANSCRIBE_AND_SUMMARIZE_RESPONSE_QUEUE_URL
        )
        assert (
            handler_args["bucket_name"]
            == sqs_worker.settings.QUEUE.TRANSCRIBE_AND_SUMMARIZE_RESULTS_BUCKET
        )

        # Verify message processor params from constants
        proc_kwargs = constructed["processor_kwargs"]
        assert (
            proc_kwargs["queue_url"]
            == sqs_worker.settings.QUEUE.TRANSCRIPTION_RESULTS_QUEUE_URL
        )
        assert proc_kwargs["max_messages"] == sqs_worker.SQSWorkerConstants.MAX_MESSAGES
        assert (
            proc_kwargs["wait_time_seconds"]
            == sqs_worker.SQSWorkerConstants.WAIT_TIME_SECONDS
        )
        assert (
            proc_kwargs["visibility_timeout"]
            == sqs_worker.SQSWorkerConstants.VISIBILITY_TIMEOUT
        )
        assert (
            proc_kwargs["polling_interval"]
            == sqs_worker.SQSWorkerConstants.POLLING_INTERVAL
        )
        assert proc_kwargs["delete_after_processing"] is True
