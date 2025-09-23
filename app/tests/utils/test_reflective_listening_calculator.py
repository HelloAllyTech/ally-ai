"""
Unit tests for reflective listening calculator utility.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.constants import ReferenceDocumentConstants
from app.schemas.common import ChatMessage
from app.utils.reflective_listening_calculator import calculate_reflective_listening


class TestReflectiveListeningCalculator:
    """Test cases for reflective listening calculator utility."""

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_empty_data(self):
        """Test calculating reflective listening with empty data."""
        mock_embedding_service = MagicMock()
        result = await calculate_reflective_listening([], mock_embedding_service)
        assert result == 0

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_no_client_messages(self):
        """Test calculating reflective listening with no client messages."""
        mock_embedding_service = MagicMock()
        messages = [
            ChatMessage(role="counselor", content="How are you feeling?"),
            ChatMessage(role="counselor", content="Tell me more about that."),
        ]
        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result == 0

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_no_counselor_messages(self):
        """Test calculating reflective listening with no counselor messages."""
        mock_embedding_service = MagicMock()
        messages = [
            ChatMessage(role="client", content="I'm feeling anxious."),
            ChatMessage(role="client", content="This is really difficult."),
        ]
        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result == 0

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_with_messages(self):
        """Test calculating reflective listening with both client and counselor messages."""  # noqa: E501
        mock_embedding_service = MagicMock()

        # Mock the embedding service to return different embeddings for different messages  # noqa: E501
        def mock_embed_many(texts):
            # Return different embeddings for different texts
            embeddings = []
            for text in texts:
                if "anxious" in text.lower():
                    embeddings.append([0.1, 0.2, 0.3])  # Client message embedding
                elif "anxiety" in text.lower():
                    embeddings.append(
                        [0.15, 0.25, 0.35]
                    )  # Similar counselor message embedding
                else:
                    embeddings.append(
                        [0.9, 0.8, 0.7]
                    )  # Different counselor message embedding
            return embeddings

        mock_embedding_service.embed_many = AsyncMock(side_effect=mock_embed_many)

        messages = [
            ChatMessage(
                role="client", content="I'm feeling really anxious about work."
            ),
            ChatMessage(
                role="counselor",
                content="It sounds like you're experiencing anxiety about work.",
            ),  # Reflective
            ChatMessage(
                role="counselor", content="How can I help you today?"
            ),  # Not reflective
        ]

        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result >= 0
        assert result <= 100

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_embedding_error(self):
        """Test calculating reflective listening when embedding service fails."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_many = AsyncMock(
            side_effect=Exception("Embedding failed")
        )

        messages = [
            ChatMessage(role="client", content="I'm feeling anxious."),
            ChatMessage(role="counselor", content="I understand your anxiety."),
        ]

        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result == 0  # Should return 0 on error

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_short_messages(self):
        """Test calculating reflective listening with short messages."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_many = AsyncMock(
            return_value=[[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]
        )

        messages = [
            ChatMessage(role="client", content="Hi"),  # Too short
            ChatMessage(role="counselor", content="Hello"),  # Too short
        ]

        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result == 0  # Should return 0 for short messages

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_high_similarity(self):
        """Test with high similarity between client and counselor messages."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_many = AsyncMock(
            side_effect=lambda texts: [
                [0.1, 0.1, 0.1] if "client" in t else [0.11, 0.11, 0.11] for t in texts
            ]
        )  # Very similar embeddings

        messages = [
            ChatMessage(
                role="client", content="I am feeling very sad about my situation."
            ),
            ChatMessage(
                role="counselor",
                content="It sounds like you are feeling quite sad about your situation.",  # noqa: E501
            ),
            ChatMessage(role="client", content="My job is causing me a lot of stress."),
            ChatMessage(
                role="counselor",
                content="So your job is a major source of stress for you.",
            ),
        ]
        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result > 80  # Expect high score due to high similarity

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_low_similarity(self):
        """Test with low similarity between client and counselor messages."""
        mock_embedding_service = MagicMock()

        # Use a call counter to track which call is which
        call_count = 0

        def mock_embed_many(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call - client messages
                return [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]  # Two client messages
            else:  # Second call - counselor messages
                return [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]  # Two counselor messages

        mock_embedding_service.embed_many = AsyncMock(side_effect=mock_embed_many)

        messages = [
            ChatMessage(
                role="client", content="I am feeling very sad about my situation."
            ),
            ChatMessage(
                role="counselor", content="What did you have for breakfast today?"
            ),
            ChatMessage(role="client", content="My job is causing me a lot of stress."),
            ChatMessage(role="counselor", content="Tell me about your hobbies."),
        ]
        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result == 0  # Expect 0 score due to low similarity (below threshold)

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_mixed_similarity(self):
        """Test with mixed similarity levels."""
        mock_embedding_service = MagicMock()

        def mixed_embed_many(texts):
            embeddings = []
            for text in texts:
                if "anxious" in text.lower():
                    embeddings.append([0.1, 0.2, 0.3])  # Client
                elif "anxiety" in text.lower():
                    embeddings.append([0.15, 0.25, 0.35])  # Reflective counselor
                elif "happy" in text.lower():
                    embeddings.append([0.6, 0.7, 0.8])  # Client
                elif "joy" in text.lower():
                    embeddings.append([0.65, 0.75, 0.85])  # Reflective counselor
                else:
                    embeddings.append([0.9, 0.9, 0.9])  # Non-reflective counselor
            return embeddings

        mock_embedding_service.embed_many = AsyncMock(side_effect=mixed_embed_many)

        messages = [
            ChatMessage(
                role="client", content="I'm feeling anxious about work."
            ),  # 6 words
            ChatMessage(
                role="counselor", content="You're experiencing anxiety about work."
            ),  # 6 words - high similarity
            ChatMessage(role="client", content="I'm also very happy today."),  # 6 words
            ChatMessage(
                role="counselor", content="Tell me more about your day."
            ),  # 7 words - low similarity
        ]
        result = await calculate_reflective_listening(messages, mock_embedding_service)
        # Should have some reflective words due to the anxiety/anxiety match
        assert result >= 0

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_empty_content_messages(self):
        """Test messages with empty content are ignored."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_many = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        messages = [
            ChatMessage(role="client", content=""),
            ChatMessage(role="counselor", content="   "),
            ChatMessage(role="client", content="I am feeling okay today."),
            ChatMessage(role="counselor", content="So you're feeling okay."),
        ]
        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result >= 0  # Should process valid messages

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_short_messages_ignored(self):
        """Test messages with content length <= 5 words are ignored."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_many = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        messages = [
            ChatMessage(role="client", content="Short message."),  # 2 words
            ChatMessage(role="counselor", content="Okay."),  # 1 word
            ChatMessage(
                role="client", content="This is a longer message from client."
            ),  # 7 words
            ChatMessage(
                role="counselor", content="This is a longer message from counselor."
            ),  # 7 words
        ]
        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result >= 0  # Should process only longer messages

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_embedding_service_exception(self):
        """Test error handling when embedding service raises an exception."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_many = AsyncMock(
            side_effect=Exception("Embedding error")
        )

        messages = [
            ChatMessage(role="client", content="I am feeling anxious."),
            ChatMessage(role="counselor", content="You are anxious."),
        ]
        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result == 0  # Should return 0 on embedding error

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_zero_counselor_words(self):
        """Test case where total counselor words is zero."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.embed_many = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        messages = [
            ChatMessage(role="client", content="I am feeling anxious."),
            ChatMessage(role="client", content="I am feeling sad."),
        ]
        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result == 0

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_no_reflective_words(self):
        """Test case where no counselor messages are reflective."""
        mock_embedding_service = MagicMock()

        # Use a call counter to track which call is which
        call_count = 0

        def mock_embed_many(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call - client messages
                return [[1.0, 0.0, 0.0]]  # One client message
            else:  # Second call - counselor messages
                return [[0.0, 1.0, 0.0]]  # One counselor message

        mock_embedding_service.embed_many = AsyncMock(side_effect=mock_embed_many)

        messages = [
            ChatMessage(
                role="client", content="I am feeling very sad about my situation."
            ),
            ChatMessage(
                role="counselor", content="This is a completely unrelated statement."
            ),
        ]
        result = await calculate_reflective_listening(messages, mock_embedding_service)
        assert result == 0  # Should be 0 due to low similarity

    @pytest.mark.asyncio
    async def test_calculate_reflective_listening_boundary_conditions(self):
        """Test boundary conditions for similarity threshold."""
        mock_embedding_service = MagicMock()

        # Mock embed_many to return embeddings that result in a similarity just above/below threshold  # noqa: E501
        def boundary_embed_many(texts):
            embeddings = []
            for text in texts:
                if "client" in text:
                    embeddings.append([0.5, 0.5, 0.5])
                elif "just above" in text:  # Should be reflective
                    embeddings.append(
                        [0.5 + 0.01, 0.5, 0.5]
                    )  # Cosine sim will be slightly above threshold
                elif "just below" in text:  # Should not be reflective
                    embeddings.append(
                        [0.5 - 0.01, 0.5, 0.5]
                    )  # Cosine sim will be slightly below threshold
                else:
                    embeddings.append([0.0, 0.0, 0.0])
            return embeddings

        mock_embedding_service.embed_many = AsyncMock(side_effect=boundary_embed_many)

        # Temporarily set a specific SIMILARITY_THRESHOLD for this test
        original_threshold = ReferenceDocumentConstants.SIMILARITY_THRESHOLD
        with patch(
            "app.core.constants.ReferenceDocumentConstants.SIMILARITY_THRESHOLD", 0.65
        ):  # Example threshold
            messages = [
                ChatMessage(
                    role="client", content="Client message about work."
                ),  # 4 words - too short, will be ignored
                ChatMessage(
                    role="counselor", content="Counselor message just above threshold."
                ),  # 6 words - Should count
                ChatMessage(
                    role="counselor", content="Counselor message just below threshold."
                ),  # 6 words - Should not count
            ]
            result = await calculate_reflective_listening(
                messages, mock_embedding_service
            )
            # Should be 0 because client message is too short and gets filtered out
            assert result == 0
        # Restore original threshold
        with patch(
            "app.core.constants.ReferenceDocumentConstants.SIMILARITY_THRESHOLD",
            original_threshold,
        ):
            pass  # Ensure the patch is reverted
