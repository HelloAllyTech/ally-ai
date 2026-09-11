"""The roadmap duplicate-detection search reports itself to the retrieval log.

Worth logging for a reason the service's own docstring states: its 0.5 threshold was
calibrated against a different embedding model at a different dimensionality, and "needs
re-calibrating against real data before it is trusted". There was no real data to calibrate
against, because nothing recorded what this search scored.

Duplicate detection also fails quietly in both directions — too low and every draft looks like
a duplicate of something, too high and the same opportunity gets filed twice — so neither
failure surfaces without the distribution.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.roadmap.roadmap_opportunity_service import RoadmapOpportunityService


@pytest.fixture
def service():
    return RoadmapOpportunityService(AsyncMock(), AsyncMock())


@pytest.mark.asyncio
async def test_reports_the_search_with_the_threshold_in_force(service):
    service.embedding_service.embed.return_value = [0.1] * 8
    service.vector_db.near_vector_search.return_value = [
        {"id": "opp-1", "product_goal": "retention", "similarity": 0.72},
        {"id": "opp-2", "product_goal": "retention", "similarity": 0.51},
    ]

    with patch(
        "app.core.roadmap.roadmap_opportunity_service.emit_retrieval_log"
    ) as emit:
        await service.search("learners abandon a track after the second session", threshold=0.5)

    kwargs = emit.call_args.kwargs
    assert kwargs["corpus"] == "roadmap_opportunities"
    assert kwargs["consumer"] == "roadmap_matcher"
    assert kwargs["min_similarity"] == 0.5
    assert kwargs["returned_count"] == 2


@pytest.mark.asyncio
async def test_does_not_mark_staff_product_text_sensitive(service):
    # Both sides of this comparison are staff-authored product descriptions. Withholding them
    # would cost the qualitative reading for nothing.
    service.embedding_service.embed.return_value = [0.1] * 8
    service.vector_db.near_vector_search.return_value = []

    with patch(
        "app.core.roadmap.roadmap_opportunity_service.emit_retrieval_log"
    ) as emit:
        await service.search("a draft opportunity")

    assert emit.call_args.kwargs["query_sensitive"] is False


@pytest.mark.asyncio
async def test_reports_an_empty_result_too(service):
    # "Nothing similar" is the answer that files a new opportunity, so it is exactly the case
    # a too-high threshold would produce wrongly.
    service.embedding_service.embed.return_value = [0.1] * 8
    service.vector_db.near_vector_search.return_value = []

    with patch(
        "app.core.roadmap.roadmap_opportunity_service.emit_retrieval_log"
    ) as emit:
        await service.search("something nobody has filed")

    assert emit.call_args.kwargs["returned_count"] == 0


@pytest.mark.asyncio
async def test_an_empty_description_searches_nothing_and_reports_nothing(service):
    with patch(
        "app.core.roadmap.roadmap_opportunity_service.emit_retrieval_log"
    ) as emit:
        assert await service.search("   ") == []

    emit.assert_not_called()
    service.vector_db.near_vector_search.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_embedding_reports_nothing(service):
    service.embedding_service.embed.side_effect = RuntimeError("embeddings down")

    with patch(
        "app.core.roadmap.roadmap_opportunity_service.emit_retrieval_log"
    ) as emit:
        with pytest.raises(Exception):
            await service.search("a draft")

    emit.assert_not_called()
