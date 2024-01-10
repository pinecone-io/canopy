from copy import deepcopy
from typing import List, Optional, Dict, Any

from canopy.knowledge_base.base import BaseKnowledgeBase
from canopy.knowledge_base.chunker import Chunker, MarkdownChunker
from canopy.knowledge_base.qdrant.constants import (
    COLLECTION_NAME_PREFIX,
    DENSE_VECTOR_NAME,
    RESERVED_METADATA_KEYS,
    SPARSE_VECTOR_NAME,
)
from canopy.knowledge_base.qdrant.converter import QdrantConverter
from canopy.knowledge_base.qdrant.utils import batched, generate_clients, sync_fallback
from canopy.knowledge_base.record_encoder import RecordEncoder, OpenAIRecordEncoder
from canopy.knowledge_base.models import (
    KBEncodedDocChunk,
    KBQueryResult,
    KBQuery,
    QueryResult,
    KBDocChunkWithScore,
    DocumentWithScore,
)
from canopy.knowledge_base.reranker import Reranker, TransparentReranker
from canopy.models.data_models import Query, Document

from qdrant_client import models as models
from qdrant_client.http.exceptions import UnexpectedResponse
from grpc import RpcError  # type: ignore
from tqdm import tqdm


class QdrantKnowledgeBase(BaseKnowledgeBase):
    """
    `QdrantKnowledgeBase` is used to store/retrieve documents using a Qdrant collection.
    Every document is chunked into multiple text snippets based on the text structure.
    Then, each chunk is encoded into a vector using an embedding model
    The resulting vectors are inserted to the Qdrant collection.
    After insertion, the `QdrantKnowledgeBase` can be queried by a textual query.
    The query will be encoded to a vector to retrieve the top-k document chunks.

    Note: Since Canopy defines its own data format,
    you cannot use a pre-existing Qdrant collection with Canopy's `QdrantKnowledgeBase`.
    The collection must be created using `knowledge_base.create_canopy_collection()`.

    Example:
        >>> from canopy.knowledge_base.knowledge_base import QdrantKnowledgeBase
        >>> kb = QdrantKnowledgeBase(collection_name="my_collection")
        >>> # Defaults to a Qdrant instance at localhost:6333
        >>> kb.create_canopy_collection()

    In any future interactions, the same collection name can be used.
    Without the need to create the collection again:

        >>> kb = QdrantKnowledgeBase(collection_name="my_collection")
    """

    _DEFAULT_COMPONENTS = {
        "record_encoder": OpenAIRecordEncoder,
        "chunker": MarkdownChunker,
        "reranker": TransparentReranker,
    }

    def __init__(
        self,
        collection_name: str,
        *,
        record_encoder: Optional[RecordEncoder] = None,
        chunker: Optional[Chunker] = None,
        reranker: Optional[Reranker] = None,
        default_top_k: int = 5,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        force_disable_check_same_thread: bool = False,
    ):
        """
        Instantiates a new `QdrantKnowledgeBase` object.

        If the collection does not exist,
        create it by calling `create_canopy_collection()`.

        Note: Canopy will add the prefix 'canopy--' to the collection name.
              You can retrieve the full collection name using
              `knowledge_base.collection_name`.

        Example:
            Create a new collection:

            >>> from canopy.knowledge_base.knowledge_base import QdrantKnowledgeBase
            >>> kb = QdrantKnowledgeBase(collection_name="my_collection")
            >>> # Defaults to a Qdrant instance at localhost:6333
            >>> kb.create_canopy_collection()

        In any future interactions, the same collection name can be used.
        Without having to create it again:

            >>> kb = QdrantKnowledgeBase(collection_name="my_collection")
        Args:
            collection_name: _description_
            record_encoder: An instance of RecordEncoder to use for encoding documents and queries.
                                              s        Defaults to OpenAIRecordEncoder.
            chunker: An instance of Chunker to use for chunking documents. Defaults to MarkdownChunker.
            reranker: An instance of Reranker to use for reranking query results. Defaults to TransparentReranker.
            default_top_k: The default number of document chunks to return per query. Defaults to 5.
            location:
                If ':memory:' - use in-memory Qdrant instance.
                If 'str' - use it as a `url` parameter.
                If 'None' - use default values for `host` and `port`.
            url: either host or str of "Optional[scheme], host, Optional[port], Optional[prefix]".
                Default: `None`
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc: If `true` - use gPRC interface whenever possible in custom methods.
            https: If `true` - use HTTPS(SSL) protocol. Default: `None`
            api_key: API key for authentication in Qdrant Cloud. Default: `None`
            prefix:
                If not `None` - add `prefix` to the REST URL path.
                Example: `service/v1` will result in `http://localhost:6333/service/v1/{qdrant-endpoint}` for REST API.
                Default: `None`
            timeout:
                Timeout for REST and gRPC API requests.
                Default: 5.0 seconds for REST and unlimited for gRPC
            host: Host name of Qdrant service. If url and host are None, set to 'localhost'.
                Default: `None`
            path: Persistence path for QdrantLocal. Default: `None`
            force_disable_check_same_thread:
                For QdrantLocal, force disable check_same_thread. Default: `False`
                Only use this if you can guarantee that you can resolve the thread safety outside QdrantClient.

        Raises:
            ValueError: If default_top_k is not a positive integer.
            TypeError: If record_encoder is not an instance of RecordEncoder.
            TypeError: If chunker is not an instance of Chunker.
            TypeError: If reranker is not an instance of Reranker.
        """  # noqa: E501
        if default_top_k < 1:
            raise ValueError("default_top_k must be greater than 0")

        self._collection_name = self._get_full_collection_name(collection_name)
        self._default_top_k = default_top_k

        if record_encoder:
            if not isinstance(record_encoder, RecordEncoder):
                raise TypeError(
                    f"record_encoder must be an instance of RecordEncoder, "
                    f"not {type(record_encoder)}"
                )
            self._encoder = record_encoder
        else:
            self._encoder = self._DEFAULT_COMPONENTS["record_encoder"]()

        if chunker:
            if not isinstance(chunker, Chunker):
                raise TypeError(
                    f"chunker must be an instance of Chunker, not {type(chunker)}"
                )
            self._chunker = chunker
        else:
            self._chunker = self._DEFAULT_COMPONENTS["chunker"]()

        if reranker:
            if not isinstance(reranker, Reranker):
                raise TypeError(
                    f"reranker must be an instance of Reranker, not {type(reranker)}"
                )
            self._reranker = reranker
        else:
            self._reranker = self._DEFAULT_COMPONENTS["reranker"]()

        self._client, self._async_client = generate_clients(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            force_disable_check_same_thread=force_disable_check_same_thread,
        )

    def verify_index_connection(self) -> None:
        """
        Verify that the knowledge base is referencing an existing Canopy collection.

        Returns:
            None

        Raises:
            RuntimeError: If the knowledge base is not referencing an existing Canopy collection.
        """  # noqa: E501
        try:
            self._client.get_collection(self.collection_name)
        except (UnexpectedResponse, RpcError, ValueError) as e:
            raise RuntimeError(
                f"Collection {self.collection_name} does not exist!"
            ) from e

    def query(
        self, queries: List[Query], global_metadata_filter: Optional[dict] = None
    ) -> List[QueryResult]:
        """
        Query the knowledge base to retrieve document chunks.

        This operation includes several steps:
        1. Encode the queries to vectors using the underlying encoder.
        2. Query the underlying Qdrant collection to retrieve the top-k chunks for each query.
        3. Rerank the results using the underlying reranker.
        4. Return the results for each query as a list of QueryResult objects.

        Args:
            queries: A list of queries to run against the knowledge base.
            global_metadata_filter: A payload filter to apply to all queries, in addition to any query-specific filters.
                                    Reference: https://qdrant.tech/documentation/concepts/filtering/
        Returns:
            A list of QueryResult objects.

        Examples:
            >>> from canopy.knowledge_base.knowledge_base import QdrantKnowledgeBase
            >>> kb = QdrantKnowledgeBase(collection_name="my_collection")
            >>> queries = [Query(text="How to make a cake"),
                           Query(text="How to make a pizza",
                                top_k=10,
                                metadata_filter={
                                                    "must": [
                                                        {"key": "website", "match": {"value": "wiki"}},
                                                    ]
                                                }
                                )]
            >>> results = kb.query(queries)
        """  # noqa: E501
        queries = self._encoder.encode_queries(queries)
        results = [self._query_collection(q, global_metadata_filter) for q in queries]
        results = self._reranker.rerank(results)

        return [
            QueryResult(
                query=r.query,
                documents=[
                    DocumentWithScore(
                        **d.dict(exclude={"values", "sparse_values", "document_id"})
                    )
                    for d in r.documents
                ],
            )
            for r in results
        ]

    @sync_fallback
    async def aquery(
        self, queries: List[Query], global_metadata_filter: Optional[dict] = None
    ) -> List[QueryResult]:
        """
        Query the knowledge base to retrieve document chunks asynchronously.

        This operation includes several steps:
        1. Encode the queries to vectors using the underlying encoder.
        2. Query the underlying Qdrant collection to retrieve the top-k chunks for each query.
        3. Rerank the results using the underlying reranker.
        4. Return the results for each query as a list of QueryResult objects.

        Args:
            queries: A list of queries to run against the knowledge base.
            global_metadata_filter: A payload filter to apply to all queries, in addition to any query-specific filters.
                                    Reference: https://qdrant.tech/documentation/concepts/filtering/
        Returns:
            A list of QueryResult objects.

        Examples:
            >>> from canopy.knowledge_base.knowledge_base import QdrantKnowledgeBase
            >>> kb = QdrantKnowledgeBase(collection_name="my_collection")
            >>> queries = [Query(text="How to make a cake"),
                           Query(text="How to make a pizza",
                                top_k=10,
                                metadata_filter={
                                                    "must": [
                                                        {"key": "website", "match": {"value": "wiki"}},
                                                    ]
                                                }
                                )]
            >>> results = await kb.aquery(queries)
        """  # noqa: E501
        # TODO: Use aencode_queries() when implemented for the defaults
        queries = self._encoder.encode_queries(queries)
        results = [
            await self._aquery_collection(q, global_metadata_filter) for q in queries
        ]
        results = self._reranker.rerank(results)

        return [
            QueryResult(
                query=r.query,
                documents=[
                    DocumentWithScore(
                        **d.dict(exclude={"values", "sparse_values", "document_id"})
                    )
                    for d in r.documents
                ],
            )
            for r in results
        ]

    def upsert(
        self,
        documents: List[Document],
        namespace: str = "",
        batch_size: int = 200,
        show_progress_bar: bool = False,
    ):
        """
        Add documents into the Qdrant collection.
        If a document with the same id already exists in the collection, it will be overwritten with the new document.
        Otherwise, a new document will be inserted.

        This operation includes several steps:
        1. Split the documents into smaller chunks.
        2. Encode the chunks to vectors.
        3. Delete any existing chunks belonging to the same documents.
        4. Upsert the chunks to the collection.

        Args:
            documents: A list of documents to upsert.
            namespace: This argument is not used by Qdrant.
            batch_size: The number of chunks (multiple piecies of text per document) to upsert in each batch.
                        Defaults to 100.
            show_progress_bar: Whether to show a progress bar while upserting the documents.


        Example:
            >>> from canopy.knowledge_base.knowledge_base import QdrantKnowledgeBase
            >>> kb = QdrantKnowledgeBase(collection_name="my_collection")
            >>> documents = [Document(id="doc1",
                                        text="This is a document",
                                        source="my_source",
                                        metadata={"website": "wiki"}),
                            Document(id="doc2",
                                     text="This is another document",
                                     source="my_source",
                                     metadata={"website": "wiki"})]
            >>> kb.upsert(documents)
        """  # noqa: E501
        for doc in documents:
            metadata_keys = set(doc.metadata.keys())
            forbidden_keys = metadata_keys.intersection(RESERVED_METADATA_KEYS)
            if forbidden_keys:
                raise ValueError(
                    f"Document with id {doc.id} contains reserved metadata keys: "
                    f"{forbidden_keys}. Please remove them and try again."
                )

        # TODO: Use achunk_documents, encode_documents when implemented for the defaults
        chunks = self._chunker.chunk_documents(documents)
        encoded_chunks = self._encoder.encode_documents(chunks)

        self._upsert_collection(encoded_chunks, batch_size, show_progress_bar)

    @sync_fallback
    async def aupsert(
        self,
        documents: List[Document],
        namespace: str = "",
        batch_size: int = 200,
        show_progress_bar: bool = False,
    ):
        """
        Add documents into the Qdrant collection asynchronously.
        If a document with the same id already exists in the collection, it will be overwritten with the new document.
        Otherwise, a new document will be inserted.

        This operation includes several steps:
        1. Split the documents into smaller chunks.
        2. Encode the chunks to vectors.
        3. Delete any existing chunks belonging to the same documents.
        4. Upsert the chunks to the collection.

        Args:
            documents: A list of documents to upsert.
            namespace: This argument is not used by Qdrant.
            batch_size: The number of chunks (multiple piecies of text per document) to upsert in each batch.
                        Defaults to 100.
            show_progress_bar: Whether to show a progress bar while upserting the documents.


        Example:
            >>> from canopy.knowledge_base.knowledge_base import QdrantKnowledgeBase
            >>> kb = QdrantKnowledgeBase(collection_name="my_collection")
            >>> documents = [Document(id="doc1",
                                        text="This is a document",
                                        source="my_source",
                                        metadata={"website": "wiki"}),
                            Document(id="doc2",
                                     text="This is another document",
                                     source="my_source",
                                     metadata={"website": "wiki"})]
            >>> await kb.aupsert(documents)
        """  # noqa: E501
        for doc in documents:
            metadata_keys = set(doc.metadata.keys())
            forbidden_keys = metadata_keys.intersection(RESERVED_METADATA_KEYS)
            if forbidden_keys:
                raise ValueError(
                    f"Document with id {doc.id} contains reserved metadata keys: "
                    f"{forbidden_keys}. Please remove them and try again."
                )

        chunks = self._chunker.chunk_documents(documents)
        encoded_chunks = self._encoder.encode_documents(chunks)

        await self._aupsert_collection(encoded_chunks, batch_size, show_progress_bar)

    def delete(self, document_ids: List[str], namespace: str = "") -> None:
        """
        Delete documents from the Qdrant collection.
        Since each document is chunked into multiple chunks, this operation will delete all chunks belonging to the given document ids.
        This operation does not raise an exception if the document does not exist.

        Args:
            document_ids: A list of document ids to delete from the Qdrant collection.
            namespace: This argument is not used by Qdrant.

        Returns:
            None

        Example:
            >>> from canopy.knowledge_base.knowledge_base import QdrantKnowledgeBase
            >>> kb = QdrantKnowledgeBase(collection_name="my_collection")
            >>> kb.delete(document_ids=["doc1", "doc2"])
        """  # noqa: E501
        self._client.delete(
            self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchAny(any=document_ids)
                    )
                ]
            ),
        )

    @sync_fallback
    async def adelete(self, document_ids: List[str], namespace: str = "") -> None:
        """
        Delete documents from the Qdrant collection asynchronously.
        Since each document is chunked into multiple chunks, this operation will delete all chunks belonging to the given document ids.
        This operation does not raise an exception if the document does not exist.

        Args:
            document_ids: A list of document ids to delete from the Qdrant collection.
            namespace: This argument is not used by Qdrant.

        Returns:
            None

        Example:
            >>> from canopy.knowledge_base.knowledge_base import QdrantKnowledgeBase
            >>> kb = QdrantKnowledgeBase(collection_name="my_collection")
            >>> await kb.adelete(document_ids=["doc1", "doc2"])
        """  # noqa: E501
        await self._async_client.delete(  # type: ignore
            self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchAny(any=document_ids)
                    )
                ]
            ),
        )

    def create_canopy_collection(
        self,
        dimension: Optional[int] = None,
        indexed_keyword_fields: List[str] = ["document_id"],
        distance: models.Distance = models.Distance.COSINE,
        shard_number: Optional[int] = None,
        sharding_method: Optional[models.ShardingMethod] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[models.HnswConfigDiff] = None,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        wal_config: Optional[models.WalConfigDiff] = None,
        quantization_config: Optional[models.QuantizationConfig] = None,
        timeout: Optional[int] = None,
        on_disk: Optional[bool] = None,
    ):
        """
        Creates a collection with the appropriate config that will be used by the QdrantKnowledgeBase.
        This is a one time set-up operation that only needs to be done once for every new Canopy service.

        Since Canopy defines its own data format with some named vectors configurations,
        you can not use a pre-existing Qdrant collection with Canopy's QdrantKnowledgeBase.

        Note: Canopy will add the prefix 'canopy--' to the collection name.
             You can retrieve the full collection name using `knowledge_base.collection_name`.

        Args:
            dimension: The dimension of the dense vectors to be used.
                       If `dimension` isn't explicitly provided,
                       Canopy would try to infer the embedding's dimension based on the configured `Encoder`
            indexed_keyword_fields: List of metadata fields to create Qdrant keyword payload index for.
                                    Defaults to ["document_id"].
            distance: Distance function to use for the vectors.
                      Defaults to COSINE.
            shard_number: Number of shards in collection. Default is 1, minimum is 1.
            sharding_method:
                Defines strategy for shard creation.
                Option `auto` (default) creates defined number of shards automatically.
                Data will be distributed between shards automatically.
                After creation, shards could be additionally replicated, but new shards could not be created.
                Option `custom` allows to create shards manually, each shard should be created with assigned
                unique `shard_key`. Data will be distributed between based on `shard_key` value.
            replication_factor:
                Replication factor for collection. Default is 1, minimum is 1.
                Defines how many copies of each shard will be created.
                Have effect only in distributed mode.
            write_consistency_factor:
                Write consistency factor for collection. Default is 1, minimum is 1.
                Defines how many replicas should apply the operation for us to consider it successful.
                Increasing this number will make the collection more resilient to inconsistencies, but will
                also make it fail if not enough replicas are available.
                Does not have any performance impact.
                Has effect only in distributed mode.
            on_disk_payload:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are indexed - remain in RAM.
            hnsw_config: Params for HNSW index
            optimizers_config: Params for optimizer
            wal_config: Params for Write-Ahead-Log
            quantization_config: Params for quantization, if None - quantization will be disabled
            timeout:
                Wait for operation commit timeout in seconds.
                If timeout is reached - request will return with service error.
            on_disk: Whethers to store vectors on disk. Defaults to None.

        """  # noqa: E501
        if dimension is None:
            try:
                encoder_dimension = self._encoder.dimension
                if encoder_dimension is None:
                    raise RuntimeError(
                        f"The selected encoder {self._encoder.__class__.__name__} does "
                        f"not support inferring the vectors' dimensionality."
                    )
                dimension = encoder_dimension
            except Exception as e:
                raise RuntimeError(
                    f"Canopy has failed to infer vectors' dimensionality using the "
                    f"selected encoder: {self._encoder.__class__.__name__}. You can "
                    f"provide the dimension manually, try using a different encoder, or"
                    f" fix the underlying error:\n{e}"
                ) from e

        try:
            self._client.get_collection(self.collection_name)

            raise RuntimeError(
                f"Collection {self.collection_name} already exists!"
                "To delete it call `knowledge_base.delete_canopy_collection()`. "
            )

        except (UnexpectedResponse, RpcError, ValueError):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=dimension, distance=distance, on_disk=on_disk
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=on_disk,
                        )
                    )
                },
                shard_number=shard_number,
                sharding_method=sharding_method,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                wal_config=wal_config,
                quantization_config=quantization_config,
                timeout=timeout,
            )

            for field in indexed_keyword_fields:
                self._client.create_payload_index(
                    self.collection_name, field_name=field, field_schema="keyword"
                )

    def list_canopy_collections(self) -> List[str]:
        collections = [
            collection.name
            for collection in self._client.get_collections().collections
            if collection.name.startswith(COLLECTION_NAME_PREFIX)
        ]
        return collections

    def delete_canopy_collection(self):
        successful = self._client.delete_collection(self.collection_name)

        if not successful:
            raise RuntimeError(f"Failed to delete collection {self.collection_name}")

    @property
    def collection_name(self) -> str:
        """
        The name of the collection the knowledge base is connected to.
        """
        return self._collection_name

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QdrantKnowledgeBase":
        """
        Create a QdrantKnowledgeBase object from a configuration dictionary.

        Args:
            config: A dictionary containing the configuration for the Qdrant knowledge base.
        Returns:
            A QdrantKnowledgeBase object.
        """  # noqa: E501

        config = deepcopy(config)
        config["params"] = config.get("params", {})
        # TODO: Add support for collection creation config for use in the CLI
        kb = cls._from_config(config)
        return kb

    @staticmethod
    def _get_full_collection_name(collection_name: str) -> str:
        if collection_name.startswith(COLLECTION_NAME_PREFIX):
            return collection_name
        else:
            return COLLECTION_NAME_PREFIX + collection_name

    def _query_collection(
        self, query: KBQuery, global_metadata_filter: Optional[dict]
    ) -> KBQueryResult:
        metadata_filter = deepcopy(query.metadata_filter)
        if global_metadata_filter is not None:
            if metadata_filter is None:
                metadata_filter = {}
            metadata_filter.update(global_metadata_filter)
        top_k = query.top_k if query.top_k else self._default_top_k

        query_params = deepcopy(query.query_params)

        query_vector = QdrantConverter.kb_query_to_search_vector(query)

        results = self._client.search(
            self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=metadata_filter,
            with_payload=True,
            **query_params,
        )

        documents: List[KBDocChunkWithScore] = []

        for result in results:
            documents.append(QdrantConverter.scored_point_to_scored_doc(result))
        return KBQueryResult(query=query.text, documents=documents)

    async def _aquery_collection(
        self, query: KBQuery, global_metadata_filter: Optional[dict]
    ) -> KBQueryResult:
        metadata_filter = deepcopy(query.metadata_filter)
        if global_metadata_filter is not None:
            if metadata_filter is None:
                metadata_filter = {}
            metadata_filter.update(global_metadata_filter)
        top_k = query.top_k if query.top_k else self._default_top_k

        query_params = deepcopy(query.query_params)

        # Use dense vector if available, otherwise use sparse vector
        query_vector = QdrantConverter.kb_query_to_search_vector(query)

        results = await self._async_client.search(  # type: ignore
            self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=metadata_filter,
            with_payload=True,
            **query_params,
        )
        documents: List[KBDocChunkWithScore] = []
        for result in results:
            documents.append(QdrantConverter.scored_point_to_scored_doc(result))
        return KBQueryResult(query=query.text, documents=documents)

    def _upsert_collection(
        self,
        encoded_chunks: List[KBEncodedDocChunk],
        batch_size: int,
        show_progress_bar: bool,
    ) -> None:
        batched_documents = batched(encoded_chunks, batch_size)
        with tqdm(
            total=len(encoded_chunks), disable=not show_progress_bar
        ) as progress_bar:
            for document_batch in batched_documents:
                batch = QdrantConverter.encoded_docs_to_points(
                    document_batch,
                )

                self._client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )

                progress_bar.update(batch_size)

    async def _aupsert_collection(
        self,
        encoded_chunks: List[KBEncodedDocChunk],
        batch_size: int,
        show_progress_bar: bool,
    ) -> None:
        batched_documents = batched(encoded_chunks, batch_size)
        with tqdm(
            total=len(encoded_chunks), disable=not show_progress_bar
        ) as progress_bar:
            for document_batch in batched_documents:
                batch = QdrantConverter.encoded_docs_to_points(
                    document_batch,
                )

                await self._async_client.upsert(  # type: ignore
                    collection_name=self.collection_name,
                    points=batch,
                )

                progress_bar.update(batch_size)

    async def close(self) -> None:
        self._client.close()
        if self._async_client:
            await self._async_client.close()
