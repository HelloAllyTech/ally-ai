from typing import List
import asyncio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.schemas.common import ChatMessage
from app.core.constants import ReferenceDocumentConstants
from app.core.embeddings.base import BaseEmbeddingService
from app.exceptions.custom_exceptions import EmbeddingFailedException
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def calculate_reflective_listening(
        chat_messages: List[ChatMessage],
        embedding_service: BaseEmbeddingService
) -> int:
    """
    Calculate reflective listening score as a percentage (0-100),
    weighted by word count of reflective counselor messages.

    This function calculates the ratio of words spoken by counselor that
    rephrase things said by client, using cosine similarity between embeddings.

    Args:
        chat_messages: List of chat messages
        embedding_service: Service to generate embeddings

    Returns:
        int: Reflective listening score as percentage (0-100)
    """
    try:
        client_messages = []
        counselor_messages = []
        counselor_word_counts = []

        for msg in chat_messages:
            text = msg.content.strip()
            if not text:
                continue

            words = text.split()
            if len(words) <= 5:
                continue

            if msg.role.lower() == 'client':
                client_messages.append(text)
            elif msg.role.lower() == 'counselor':
                counselor_messages.append(text)
                counselor_word_counts.append(len(words))

        if not client_messages or not counselor_messages:
            logger.debug("No client or counselor messages found for reflective listening calculation")
            return 0

        # embed in parallel
        client_embeddings_task = embedding_service.embed_many(client_messages)
        counselor_embeddings_task = embedding_service.embed_many(counselor_messages)
        client_embeddings, counselor_embeddings = await asyncio.gather(
            client_embeddings_task,
            counselor_embeddings_task
        )

        client_arr = np.array(client_embeddings)
        counselor_arr = np.array(counselor_embeddings)

        sim_matrix = cosine_similarity(counselor_arr, client_arr)
        max_sims = np.max(sim_matrix, axis=1)

        # sum word counts of messages that exceed threshold
        reflective_words = sum(
            wc for wc, sim in zip(counselor_word_counts, max_sims) if
            sim > ReferenceDocumentConstants.SIMILARITY_THRESHOLD
        )
        total_counselor_words = sum(counselor_word_counts)

        if total_counselor_words == 0:
            return 0

        reflective_percentage = (reflective_words / total_counselor_words) * 100
        reflective_score = min(100, max(0, round(reflective_percentage)))

        return reflective_score

    except EmbeddingFailedException as e:
        logger.error(f"Embedding service failed during reflective listening calculation: {str(e)}")
        return 0
    except Exception as e:
        logger.error(f"Error calculating reflective listening: {str(e)}")
        return 0
