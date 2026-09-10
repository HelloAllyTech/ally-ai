from asyncio import Semaphore
from typing import Any, Dict, List, Optional

import weaviate.classes as wvc
from weaviate.classes.query import Filter
from weaviate.client import WeaviateAsyncClient
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.collections.classes.internal import QueryReturn
from weaviate.exceptions import (
    AuthenticationFailedException,
    WeaviateConnectionError,
    WeaviateInsertManyAllFailedError,
)

from app.core.config import settings
from app.core.embeddings.base import BaseEmbeddingService
from app.core.vector_db.base import VectorDB
from app.core.vector_db.constants import VectorDBCollectionNames
from app.exceptions.custom_exceptions import (
    DocumentNotFoundException,
    EmbeddingFailedException,
    VectorDBDeleteFailedException,
    VectorDBFetchFailedException,
    VectorDBInsertFailedException,
    VectorDBSearchFailedException,
    VectorDBUpdateFailedException,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _build_property_filter(filters: Optional[Dict[str, Any]]):
    """
    Build a Weaviate filter from a plain {property: value} dict, ANDing the conditions.

    A LIST, TUPLE or SET value means "any of these", built as an OR of equalities rather
    than as `contains_any`. Same primitive `delete_by_filter` already trusts for exact
    document_id matching, so a set-membership filter is exactly as precise as the delete
    that pairs with it — `contains_any` on a word-tokenised TEXT property would be a
    second, subtly different notion of equality.

    This exists because "any of these ids" has to be a PRE-filter. Applied after the
    search instead, a restriction cannot merely narrow the result — it starves it: the
    engine ranks the whole collection, hands back its global top-k, and the passage that
    was the best match *within the allowed set* is simply not in that window. A scoped
    search that silently returns nothing (or, worse, gets widened by a caller who
    compensates with a bigger k) is the failure this prevents.

    An EMPTY collection raises rather than being dropped. Dropping it would widen the
    query to the whole collection, the opposite of what the caller asked for, and is
    precisely how a corpus boundary leaks — one consumer's material answering another
    consumer's question. `None` is different and is still skipped: that means "no
    restriction on this property" and is how an optional filter is expressed.
    """
    if not filters:
        return None

    conditions = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            allowed = list(value)
            if not allowed:
                raise ValueError(
                    f"Filter '{key}' was given an empty set of allowed values. That "
                    "matches nothing, and treating it as no filter would widen the "
                    "search instead of narrowing it."
                )
            conditions.append(
                Filter.any_of([Filter.by_property(key).equal(v) for v in allowed])
            )
        else:
            conditions.append(Filter.by_property(key).equal(value))

    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else Filter.all_of(conditions)


class WeaviateDB(VectorDB):
    #: Objects fetched per page by `update_properties_by_filter`. Weaviate has no
    #: update-by-filter, so that sweep is a paged read plus one update per object; a
    #: failure costs one page of retries.
    UPDATE_PAGE_SIZE = 200

    #: Refuse rather than rewrite beyond this many objects. Comfortably above any
    #: legitimate caller (ally-be caps a document at 3000 chunks), so reaching it means
    #: the filter matched far more than the caller meant.
    UPDATE_MAX_OBJECTS = 10_000

    def __init__(
        self, client: WeaviateAsyncClient, embedding_service: BaseEmbeddingService
    ) -> None:
        self.embedding_service = embedding_service
        self._semaphore = Semaphore(settings.WEAVIATE.CONCURRENT_REQUESTS)
        super().__init__(client)

    async def similarity_search(
        self, vector: List[float], top_k: int = 1
    ) -> QueryReturn:
        """
        Perform a similarity search in the vector database using the provided vector.

        This method retrieves the "CONVERSATIONS" collection from the
        Weaviate client and executes a near vector query to find the
        top_k most similar items. The search request includes
        metadata retrieval with a certainty score.

        Parameters:
            vector (List[float]): The embedding vector to search for similar items.
            top_k (int, optional): The maximum number of similar results to return.
            Defaults to 1.

        Returns:
            QueryReturn: The result of the similarity search, including the matching
            items and associated metadata.

        Raises:
            VectorDBSearchFailedException: If a connection or authentication error
            occurs while querying Weaviate.
        """
        collection = self.client.collections.get(VectorDBCollectionNames.CONVERSATIONS)

        try:
            async with self._semaphore:
                return await collection.query.near_vector(
                    near_vector=vector,
                    limit=top_k,
                    return_metadata=wvc.query.MetadataQuery(certainty=True),
                )

        except WeaviateConnectionError as e:
            logger.exception(f"Weaviate connection error: {type(e).__name__}")
            raise VectorDBSearchFailedException(
                "Weaviate connection error. Please try again later."
            ) from e

        except AuthenticationFailedException as e:
            logger.exception(f"Weaviate authentication error: {type(e).__name__}")
            raise VectorDBSearchFailedException(
                "Weaviate authentication failed. Please try again later."
            ) from e

    async def fetch_relevant_conversations(
        self, query: str, top_k: int = 1
    ) -> QueryReturn:
        """
        Fetches the most relevant conversations from Weaviate for the given query.

        This function generates an embedding for the provided query
        using the embedding service, and then performs a similarity
        search in Weaviate to retrieve the top matching conversations.

        Parameters:
            query (str): The query string to search for relevant conversations.
            top_k (int, optional): The maximum number of conversations to return.
            Defaults to 1.

        Returns:
            QueryReturn: The result of the similarity search, which includes the most
            relevant conversations and associated metadata.

        Raises:
            VectorDBFetchFailedException: If the embedding generation fails or if the
            similarity search encounters an error.
        """
        try:
            vector = await self.embedding_service.embed(query)

            logger.info("Fetching relevant conversations for query")
            return await self.similarity_search(vector, top_k)

        except EmbeddingFailedException as e:
            raise VectorDBFetchFailedException(
                "Embedding failed. Please try again later."
            ) from e

        except VectorDBSearchFailedException as e:
            raise VectorDBFetchFailedException(
                "Weaviate search failed. Please try again later."
            ) from e

    async def create_document(
        self,
        collection_name: str,
        document_data: Dict[str, Any],
        vector: List[float],
        document_id: str,
    ) -> str:
        """
        Create a new document in the Weaviate database.

        Parameters:
            collection_name (str): Name of the collection to store the document in.
            document_data (Dict[str, Any]): Data to be stored in the document.
            vector (List[float]): The embedding vector for the document.
            document_id (str): UUID to use as the document ID.

        Returns:
            str: The UUID of the created document.

        Raises:
            VectorDBInsertFailedException: If document insertion fails.
        """
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)

            # Create the document with the specified UUID
            async with self._semaphore:
                result_id = await collection.data.insert(
                    properties=document_data, vector=vector, uuid=document_id
                )

            # Return the UUID of the created document
            return str(result_id)

        except Exception as e:
            logger.exception(f"Failed to create document: {type(e).__name__}")
            raise VectorDBInsertFailedException("Failed to create document")

    async def create_documents_bulk(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Insert many objects in one round trip. See VectorDB.create_documents_bulk.

        Two things worth knowing about `insert_many`:

        * It does NOT raise on per-object failure. The returned BatchObjectReturn
          carries an
          `errors` mapping keyed by the object's INDEX in the submitted list, and
          callers that ignore it get silent data loss. So the errors are translated back
          to ids here, which is the only identifier the caller reasons in.
        * It is an upsert only in the sense that a duplicate UUID is reported as a
          per-object
          error, not silently merged. Callers re-indexing a document therefore delete
          the old generation first (delete_by_filter) rather than relying on overwrite
          semantics.

        An object with an empty vector is rejected before the request rather than
        written: these collections are created with vectorizer_config=none, so Weaviate
        will not generate one, and a vector-less object is accepted, stored, and then
        never returned by any similarity search. That failure is invisible until someone
        asks the question the chunk would have answered, which is exactly the kind of
        silence worth spending a guard on.
        """
        if not documents:
            return {"succeeded": [], "failed": []}

        failed: List[Dict[str, str]] = []
        writable: List[Dict[str, Any]] = []

        for doc in documents:
            doc_id = str(doc.get("id", ""))
            if not doc.get("vector"):
                failed.append(
                    {
                        "id": doc_id,
                        "error": (
                            "missing embedding vector — the object would be stored "
                            "but never retrievable, since this collection has no "
                            "vectorizer"
                        ),
                    }
                )
                continue
            writable.append(doc)

        if not writable:
            return {"succeeded": [], "failed": failed}

        try:
            collection = self.client.collections.get(collection_name)

            batch = [
                wvc.data.DataObject(
                    uuid=str(doc["id"]),
                    properties=doc["properties"],
                    vector=doc["vector"],
                )
                for doc in writable
            ]

            async with self._semaphore:
                result = await collection.data.insert_many(batch)

        except WeaviateInsertManyAllFailedError as e:
            # The client raises instead of returning when EVERY object fails, which
            # would otherwise turn a per-object result into a whole-request exception.
            # Translate it back into the per-object contract: the caller tracks progress
            # per chunk, so an all-failed batch has to look like N failures, not one
            # unexplained error, or its retry bookkeeping silently loses the batch.
            logger.error(f"Bulk insert: all {len(writable)} object(s) failed")
            failed.extend({"id": str(doc["id"]), "error": str(e)} for doc in writable)
            return {"succeeded": [], "failed": failed}

        except Exception as e:
            logger.exception(f"Bulk insert failed: {type(e).__name__}")
            raise VectorDBInsertFailedException("Failed to bulk insert documents")

        # `errors` is keyed by position in the submitted batch; map it back to ids so
        # the caller never has to reason about our batching.
        errors = getattr(result, "errors", None) or {}
        succeeded: List[str] = []
        for index, doc in enumerate(writable):
            doc_id = str(doc["id"])
            error = errors.get(index)
            if error is None:
                succeeded.append(doc_id)
            else:
                message = getattr(error, "message", None) or str(error)
                logger.warning(f"Bulk insert rejected one object in {collection_name}")
                failed.append({"id": doc_id, "error": message})

        return {"succeeded": succeeded, "failed": failed}

    async def delete_by_filter(
        self,
        collection_name: str,
        filters: Dict[str, Any],
    ) -> int:
        """
        Delete every object matching property equality filters. See
        VectorDB.delete_by_filter.

        The empty-filter guard is a ValueError raised BEFORE touching Weaviate, not a
        silent no-op: `delete_many` with a match-everything filter would empty the
        collection, and a caller that passed an accidentally-empty dict (a None document
        id stringified away, say) deserves an exception rather than a successful-looking
        wipe.
        """
        conditions = [
            Filter.by_property(key).equal(value)
            for key, value in (filters or {}).items()
            if value is not None
        ]
        if not conditions:
            raise ValueError(
                "delete_by_filter requires at least one non-null filter; an empty "
                "filter would match every object in the collection"
            )

        try:
            collection = self.client.collections.get(collection_name)
            where = conditions[0] if len(conditions) == 1 else Filter.all_of(conditions)

            async with self._semaphore:
                result = await collection.data.delete_many(where=where)

            # `failed` is non-zero when some matches could not be removed. Reported
            # rather than swallowed: the caller re-indexing a document needs to know the
            # old generation may still be retrievable, or it will serve duplicate
            # citations from two versions.
            failed_count = int(getattr(result, "failed", 0) or 0)
            if failed_count:
                logger.error(
                    f"delete_by_filter left {failed_count} object(s) "
                    f"in {collection_name}"
                )
            return int(getattr(result, "successful", 0) or 0)

        except Exception as e:
            logger.exception(f"delete_by_filter failed: {type(e).__name__}")
            raise VectorDBDeleteFailedException("Failed to delete documents by filter")

    async def get_document_by_id(
        self, collection_name: str, document_id: str, include_vector: bool = True
    ) -> Dict[str, Any]:
        """
        Get a document by its ID from the Weaviate database.

        Parameters:
            collection_name (str): Name of the collection to retrieve the document from.
            document_id (str): ID of the document to retrieve.
            include_vector (bool): Whether to include the vector in the response.

        Returns:
            Dict[str, Any]: The document data.

        Raises:
            DocumentNotFoundException: If the document is not found.
        """
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)

            # Get the document
            async with self._semaphore:
                result = await collection.query.fetch_objects(
                    limit=1,
                    filters=wvc.query.Filter.by_id().equal(document_id),
                    include_vector=include_vector,
                )

            if not result.objects or len(result.objects) == 0:
                raise DocumentNotFoundException(
                    f"Document with ID {document_id} not found"
                )

            # Extract document properties
            obj = result.objects[0]
            document = {"id": obj.uuid, **obj.properties}

            # Include vector if requested and available
            if include_vector and obj.vector:
                document["vector"] = obj.vector

            return document

        except DocumentNotFoundException as e:
            # Re-raise DocumentNotFoundException
            raise e
        except Exception as e:
            logger.exception(f"Failed to get document: {type(e).__name__}")
            raise DocumentNotFoundException(f"Document with ID {document_id} not found")

    async def update_properties_by_filter(
        self,
        collection_name: str,
        filters: Dict[str, Any],
        properties: Dict[str, Any],
    ) -> int:
        """
        Merge properties into every object matching a filter. See
        VectorDB.update_properties_by_filter.

        Weaviate has no update-by-filter, so this collects the matching ids and then
        updates them one at a time. `data.update` is a MERGE — properties left out are
        untouched — and no vector is passed, so the embedding is preserved.

        Ids are collected FIRST, and paged with `offset` rather than the `after` cursor.
        Both parts are deliberate. The cursor is the paging mode that cannot be joined
        to a filter, and every use of this method is filtered by definition. Collecting
        before writing means the page boundaries are decided against one consistent read
        rather than against a set being mutated underneath the sweep; page-then-write
        walks a moving target, and an object skipped that way silently keeps the
        properties it used to have.

        `UPDATE_MAX_OBJECTS` bounds the collection phase. It sits well above any
        legitimate caller — ally-be caps a document at 3000 chunks — so reaching it
        means the filter matched something far wider than intended, and refusing beats
        rewriting a corpus.
        """
        conditions = [
            Filter.by_property(key).equal(value)
            for key, value in (filters or {}).items()
            if value is not None
        ]
        if not conditions:
            raise ValueError(
                "update_properties_by_filter requires at least one non-null filter; an "
                "empty filter would rewrite every object in the collection"
            )
        if not properties:
            raise ValueError(
                "update_properties_by_filter requires at least one property to write"
            )

        where = conditions[0] if len(conditions) == 1 else Filter.all_of(conditions)
        updated = 0

        try:
            collection = self.client.collections.get(collection_name)

            ids: List[str] = []
            offset = 0
            while True:
                async with self._semaphore:
                    page = await collection.query.fetch_objects(
                        limit=self.UPDATE_PAGE_SIZE,
                        offset=offset,
                        filters=where,
                        return_properties=[],
                    )
                if not page.objects:
                    break

                ids.extend(str(obj.uuid) for obj in page.objects)
                if len(page.objects) < self.UPDATE_PAGE_SIZE:
                    break
                if len(ids) >= self.UPDATE_MAX_OBJECTS:
                    raise ValueError(
                        f"update_properties_by_filter matched more than "
                        f"{self.UPDATE_MAX_OBJECTS} objects in {collection_name}; "
                        "refusing to continue"
                    )
                offset += self.UPDATE_PAGE_SIZE

            for object_id in ids:
                async with self._semaphore:
                    await collection.data.update(uuid=object_id, properties=properties)
                updated += 1

            return updated

        except Exception as e:
            logger.exception(
                f"update_properties_by_filter failed after {updated} update(s): "
                f"{type(e).__name__}"
            )
            # The count is logged before raising: a partial update leaves the collection
            # in a mixed state, and the caller needs to know a retry is a resume rather
            # than a fresh start.
            raise VectorDBUpdateFailedException("Failed to update documents by filter")

    async def update_document(
        self,
        collection_name: str,
        document_id: str,
        document_data: Dict[str, Any],
        vector: List[float],
    ) -> None:
        """
        Update an existing document in the Weaviate database.

        Parameters:
            collection_name (str): Name of the collection containing the document.
            document_id (str): ID of the document to update.
            document_data (Dict[str, Any]): Updated data for the document.
            vector (List[float]): Updated embedding vector.

        Raises:
            VectorDBUpdateFailedException: If the document update fails.
        """
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)

            # Update the document directly without checking existence
            async with self._semaphore:
                await collection.data.update(
                    uuid=document_id, properties=document_data, vector=vector
                )

        except Exception as e:
            logger.exception(f"Failed to update document: {type(e).__name__}")
            raise VectorDBUpdateFailedException("Failed to update document")

    async def delete_document(self, collection_name: str, document_id: str) -> None:
        """
        Delete a document from the Weaviate database.

        Parameters:
            collection_name (str): Name of the collection containing the document.
            document_id (str): ID of the document to delete.

        Raises:
            VectorDBDeleteFailedException: If the document deletion fails.
        """
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)

            # Delete the document
            async with self._semaphore:
                await collection.data.delete_by_id(document_id)

        except Exception as e:
            logger.exception(f"Failed to delete document: {type(e).__name__}")
            raise VectorDBDeleteFailedException("Failed to delete document")

    async def list_document_ids(
        self,
        collection_name: str,
        limit: int = 200,
        after: Optional[str] = None,
    ) -> List[str]:
        """
        One page of object UUIDs. See VectorDB.list_document_ids.

        `return_properties=[]` keeps this to ids on the wire — a reconciliation sweep
        over a whole collection has no use for properties, and fetching them would
        multiply the payload for nothing.
        """
        try:
            collection = self.client.collections.get(collection_name)

            async with self._semaphore:
                result = await collection.query.fetch_objects(
                    limit=limit,
                    after=after,
                    return_properties=[],
                )

            return [str(obj.uuid) for obj in result.objects]

        except Exception as e:
            logger.exception(f"list_document_ids failed: {type(e).__name__}")
            raise VectorDBSearchFailedException("Failed to list document ids")

    async def search_documents(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for documents in the vector database based on a query string.

        Args:
            collection_name: Name of the collection to search in
            query: Query string to search for
            limit: Maximum number of results to return
            filters: Optional filters to apply to the search
            include_vector: Whether to include the vector in the response

        Returns:
            Dictionary containing the search results
        """
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)

            # Generate vector embedding for the query
            vector = await self.embedding_service.embed(query)

            # Build query filters if provided
            query_filters = None
            if filters:
                # Check if we have document IDs to filter by
                if "id" in filters:
                    doc_ids = filters.pop("id")

                    # Handle single ID or list of IDs
                    if isinstance(doc_ids, list):
                        if len(doc_ids) == 1:
                            id_filter = Filter.by_id().equal(doc_ids[0])
                        else:
                            # For multiple IDs, create OR conditions
                            id_conditions = [
                                Filter.by_id().equal(doc_id) for doc_id in doc_ids
                            ]
                            id_filter = Filter.any_of(id_conditions)
                    else:
                        id_filter = Filter.by_id().equal(doc_ids)

                    # Start with ID filter
                    query_filters = id_filter

                # Process other filters
                if filters:
                    filter_conditions = []
                    for key, value in filters.items():
                        if isinstance(value, list):
                            # For list values (like tags), check if any value matches
                            # Create individual filter conditions for each value
                            or_conditions = [
                                Filter.by_property(key).equal(item) for item in value
                            ]
                            if or_conditions:
                                # For a single item, just add it directly
                                if len(or_conditions) == 1:
                                    filter_conditions.append(or_conditions[0])
                                else:
                                    # For multiple items, use a list of conditions
                                    # with Filter.any_of
                                    filter_conditions.append(
                                        Filter.any_of(or_conditions)
                                    )
                        else:
                            # For single values
                            filter_conditions.append(
                                Filter.by_property(key).equal(value)
                            )

                    # Combine all conditions with AND
                    if filter_conditions:
                        property_filter = (
                            filter_conditions[0]
                            if len(filter_conditions) == 1
                            else Filter.all_of(filter_conditions)
                        )

                        # Combine with ID filter if it exists
                        if query_filters:
                            query_filters = Filter.all_of(
                                [query_filters, property_filter]
                            )
                        else:
                            query_filters = property_filter

            total = 0
            categories = {}
            # Single aggregation call that gets both total count and category breakdown
            async with self._semaphore:
                # Use near_vector for aggregation (more reliable than near_text)
                agg_result = await collection.aggregate.near_vector(
                    near_vector=vector,
                    filters=query_filters,
                    distance=settings.REFERENCE_DOCUMENTS_DISTANCE_THRESHOLD,
                    group_by="category",
                )
                # Get category breakdown
                categories = {}
                for group in agg_result.groups:
                    category_name = group.grouped_by.value
                    category_count = group.total_count
                    categories[category_name] = category_count
                    total += category_count

            # Execute the main query with the appropriate parameters
            async with self._semaphore:
                result = await collection.query.near_vector(
                    near_vector=vector,
                    limit=limit,
                    filters=query_filters,
                    include_vector=include_vector,
                    distance=settings.REFERENCE_DOCUMENTS_DISTANCE_THRESHOLD,
                    return_metadata=MetadataQuery(
                        distance=True
                    ),  # Correctly specify metadata to return
                )
            # Process the results
            documents = []
            for obj in result.objects:
                document = {
                    "id": str(obj.uuid),  # Convert UUID to string
                    **obj.properties,
                }

                # Add distance score (similarity score)
                if obj.metadata and obj.metadata.distance is not None:
                    document["score"] = (
                        1.0 - obj.metadata.distance
                    )  # Convert distance to similarity score
                else:
                    document["score"] = None

                if include_vector and obj.vector:
                    document["vector"] = obj.vector
                documents.append(document)

            # Return the search results
            return {
                "documents": documents,
                "total": total if total is not None else len(documents),
                "categories": categories,
            }
        except Exception as e:
            logger.exception(f"Failed to search documents: {type(e).__name__}")
            raise VectorDBSearchFailedException("Failed to search documents")

    @staticmethod
    def _build_condition(entry: Dict[str, Any]):
        """
        Turn one declarative condition from `any_of` into a Weaviate filter.

        Deliberately a CLOSED set of two operators rather than a general expression
        language. This is reached from the audience filter on the request path, and the
        one failure that must be impossible is a condition that quietly evaluates to
        "everything" — so an entry naming no property, or an operator this does not
        implement, raises instead of degrading into a match-all.
        """
        prop = entry.get("property")
        if not prop:
            raise ValueError("an any_of condition must name a property")

        if "equal" in entry:
            return Filter.by_property(prop).equal(entry["equal"])
        if "contains_any" in entry:
            values = entry["contains_any"] or []
            if not values:
                # `contains_any([])` is accepted by the client and matches nothing, but
                # spelling it out here keeps the intent readable at the call site.
                raise ValueError(
                    f"contains_any on '{prop}' was given no values; pass an empty "
                    "any_of list to mean 'match nothing' instead"
                )
            return Filter.by_property(prop).contains_any(values)

        raise ValueError(
            f"unsupported any_of condition for '{prop}': expected 'equal' or "
            "'contains_any'"
        )

    async def near_vector_search(
        self,
        collection_name: str,
        vector: List[float],
        limit: int = 10,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        any_of: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Similarity search from a caller-supplied vector. See
        VectorDB.near_vector_search.

        Weaviate returns DISTANCE; with the cosine metric, similarity = 1 - distance.
        The conversion happens here so callers only ever reason in similarity, which is
        what the standalone roadmap app's threshold (0.5) was expressed in.
        """
        # An `any_of` that came in EMPTY is answered before any I/O. It means the caller
        # computed a disjunction that nothing can satisfy — an audience matching no
        # documents, say — and the one thing that must not happen is falling through to
        # an unfiltered search. `None` (no disjunction asked for) is a different case
        # continues below.
        if any_of is not None and not any_of:
            return []

        try:
            collection = self.client.collections.get(collection_name)

            query_filters = _build_property_filter(filters)

            if any_of:
                disjunction = Filter.any_of(
                    [self._build_condition(entry) for entry in any_of]
                )
                query_filters = (
                    disjunction
                    if query_filters is None
                    else Filter.all_of([query_filters, disjunction])
                )

            # A similarity floor is a distance ceiling.
            max_distance = 1.0 - min_similarity

            async with self._semaphore:
                result = await collection.query.near_vector(
                    near_vector=vector,
                    limit=limit,
                    filters=query_filters,
                    distance=max_distance,
                    return_metadata=MetadataQuery(distance=True),
                )

            hits: List[Dict[str, Any]] = []
            for obj in result.objects:
                distance = getattr(obj.metadata, "distance", None)
                similarity = 1.0 - distance if distance is not None else 0.0
                hits.append(
                    {"id": str(obj.uuid), "similarity": similarity, **obj.properties}
                )
            return hits

        except Exception as e:
            logger.exception(f"near_vector_search failed: {type(e).__name__}")
            raise VectorDBSearchFailedException("Failed to run similarity search")
