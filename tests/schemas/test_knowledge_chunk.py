"""Tests for the knowledge-corpus wire schemas.

Only the audience shapes are covered here: the rest of these models are plain field
mirrors of ally-be rows, whereas the audience is a rule about who may read what, and its
failure mode is over-sharing rather than an exception.
"""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.knowledge_chunk import (
    AudienceSelector,
    KnowledgeChunkItem,
    KnowledgeChunkSearchRequest,
)


class TestAudienceSelector:
    def test_tenant_audience_maps_to_the_service_value_object(self):
        tenant_id = uuid4()

        audience = AudienceSelector(tenant_id=tenant_id).to_chunk_audience()

        assert audience.tenant_id == str(tenant_id)
        assert audience.include_global is True
        assert audience.ignore_targeting is False

    def test_unrestricted_maps_to_no_filtering(self):
        audience = AudienceSelector(unrestricted=True).to_chunk_audience()

        assert audience.ignore_targeting is True
        assert audience.to_any_of() is None

    def test_an_audience_that_matches_nothing_is_refused(self):
        with pytest.raises(ValidationError):
            AudienceSelector(tenant_id=None, include_global=False)

    def test_unrestricted_survives_the_empty_audience_check(self):
        """`unrestricted` names neither an organisation nor the global corpus."""
        assert AudienceSelector(
            tenant_id=None, include_global=False, unrestricted=True
        ).unrestricted


class TestKnowledgeChunkAudienceDefaults:
    def test_an_indexed_chunk_defaults_to_reaching_nobody(self):
        """
        The closed default. A caller that forgets the audience indexes a passage nobody
        retrieves — visible and fixable — rather than one everybody retrieves.
        """
        item = KnowledgeChunkItem(
            chunk_id=uuid4(), document_id=uuid4(), text="a passage"
        )

        assert item.is_global is False
        assert item.tenant_ids == []

    def test_the_admin_search_defaults_to_unrestricted(self):
        """
        The deliberate asymmetry with the answering agent.

        This endpoint backs the corpus browser and retrieval preview, which answer "what
        is indexed", not "what may this worker see".
        """
        assert KnowledgeChunkSearchRequest(query="q").audience.unrestricted is True
