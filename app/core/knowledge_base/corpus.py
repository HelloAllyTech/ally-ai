"""Which corpus a knowledge request is about, and which collection holds it.

One pipeline (extract → chunk → index → retrieve → cite), several consumers. The corpus
is the only thing they differ by, so it is the one thing they declare — and it resolves
to a COLLECTION rather than to a filter value.

That resolution is the whole point. Both corpora are Ally-global and non-confidential,
so a leak here would not expose anyone's data; it would do something subtler and
harder to notice, which is answer a health worker's clinical question out of material
written to ground a fictional character. That reads as good clinical prose. The
reasons for a collection per corpus rather than one collection plus a scope
argument are recorded in
migration 005 and in KnowledgeChunkProperties: a similarity threshold only means
something against one distribution and chunk size is per corpus; filtered ANN search is
weaker than unfiltered; and re-chunking one corpus must not touch the index another is
being served from.
"""

from enum import Enum

from app.core.vector_db.constants import VectorDBCollectionNames


class KbCorpus(str, Enum):
    """Mirrors ally-be's KbCorpus. The wire values are the ones stored in
    kb_documents.corpus, so the two must not drift."""

    WHATSAPP_QA = "whatsapp_qa"
    CHARACTER_LIBRARY = "character_library"


CORPUS_COLLECTIONS: dict[KbCorpus, str] = {
    KbCorpus.WHATSAPP_QA: VectorDBCollectionNames.KNOWLEDGE_CHUNKS,
    KbCorpus.CHARACTER_LIBRARY: VectorDBCollectionNames.CHARACTER_CHUNKS,
}


def collection_for(corpus: KbCorpus) -> str:
    """The collection holding one corpus's chunks.

    A KeyError here is a programming error, not a request error: every KbCorpus member
    must have a collection and a migration creating it.
    """
    return CORPUS_COLLECTIONS[corpus]
