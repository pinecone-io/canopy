from itertools import islice
import os
from copy import deepcopy
from typing import List, Optional, Dict, Any, Union

from canopy.knowledge_base.base import BaseKnowledgeBase
from canopy.knowledge_base.chunker import Chunker, MarkdownChunker
from canopy.knowledge_base.qdrant.constants import (
    COLLECTION_NAME_PREFIX,
    DENSE_VECTOR,
    RESERVED_METADATA_KEYS,
    SPARSE_VECTOR,
)
from canopy.knowledge_base.qdrant.converter import QdrantConverter
from canopy.knowledge_base.record_encoder import RecordEncoder, OpenAIRecordEncoder
from canopy.knowledge_base.models import (
    KBQueryResult,
    KBQuery,
    QueryResult,
    KBDocChunkWithScore,
    DocumentWithScore,
)
from canopy.knowledge_base.reranker import Reranker, TransparentReranker
from canopy.models.data_models import Query, Document

from qdrant_client import QdrantClient, models as models
from qdrant_client.http.exceptions import UnexpectedResponse
from grpc import RpcError  # type: ignore
from tqdm import tqdm


def batched(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.

    """
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class QdrantKnowledgeBase(BaseKnowledgeBase):
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
        collection_create_params: Optional[Dict[str, Any]] = None,
    ):
        if default_top_k < 1:
            raise ValueError("default_top_k must be greater than 0")

        self._collection_name = self._get_full_index_name(collection_name)
        self._default_top_k = default_top_k
        self._collection_crate_params = collection_create_params

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

        # Normally, index creation params are passed directly to the `.create_canopy_index()` method.  # noqa: E501
        # However, when KnowledgeBase is initialized from a config file, these params
        # would be set by the `KnowledgeBase.from_config()` constructor.
        self._collection_params: Dict[str, Any] = {}

        # The index object is initialized lazily, when the user calls `connect()` or
        # `create_canopy_index()`
        self._client = QdrantClient(
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
        )

    @property
    def _connection_error_msg(self) -> str:
        return (
            f"KnowledgeBase is not connected to index {self.collection_name}, "
            f"Please call knowledge_base.connect(). "
        )

    def verify_index_connection(self) -> None:
        try:
            self._client.get_collection(self._collection_name)
        except (UnexpectedResponse, RpcError, ValueError) as e:
            raise RuntimeError(
                f"Collection {self._collection_name} does not exist!"
            ) from e

    def create_canopy_collection(
        self,
        dimension: Optional[int] = None,
        distance: models.Distance = models.Distance.COSINE,
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[models.HnswConfigDiff] = None,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        wal_config: Optional[models.WalConfigDiff] = None,
        quantization_config: Optional[models.QuantizationConfig] = None,
        init_from: Optional[models.InitFrom] = None,
        on_disk: Optional[bool] = None,
    ):
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
            self._client.get_collection(self._collection_name)

            raise RuntimeError(
                f"Collection {self.collection_name} already exists!"
                "If you wish to delete it call `knowledge_base.delete_collection()`. "
            )

        except (UnexpectedResponse, RpcError, ValueError):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=dimension, distance=distance, on_disk=on_disk
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=on_disk,
                        )
                    )
                },
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                wal_config=wal_config,
                quantization_config=quantization_config,
                init_from=init_from,
            )

    @staticmethod
    def _get_full_index_name(index_name: str) -> str:
        if index_name.startswith(COLLECTION_NAME_PREFIX):
            return index_name
        else:
            return COLLECTION_NAME_PREFIX + index_name

    @property
    def collection_name(self) -> str:
        """
        The name of the index the knowledge base is connected to.
        """
        return self._collection_name

    def query(
        self, queries: List[Query], global_metadata_filter: Optional[dict] = None
    ) -> List[QueryResult]:
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

        # Use dense vector if available, otherwise use sparse vector
        query_vector: Union[models.NamedSparseVector, models.NamedVector] = (
            models.NamedVector(name=DENSE_VECTOR, vector=query.values)
            if query.values is not None
            else models.NamedSparseVector(
                name=SPARSE_VECTOR,
                vector=models.SparseVector(
                    indices=query.sparse_values["indices"],  # type: ignore
                    values=query.sparse_values["values"],  # type: ignore
                ),
            )
        )

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

    def upsert(
        self,
        documents: List[Document],
        namespace: str = "",
        batch_size: int = 200,
        show_progress_bar: bool = False,
    ):
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

    def delete(self, document_ids: List[str], namespace: str = "") -> None:
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

    def list_canopy_collections(self) -> List[str]:
        collections = [
            collection.name
            for collection in self._client.get_collections().collections
            if collection.name.startswith(COLLECTION_NAME_PREFIX)
        ]
        return collections

    def delete_collection(self):
        successful = self._client.delete_collection(self._collection_name)

        if not successful:
            raise RuntimeError(f"Failed to delete collection {self.collection_name}")

    async def aquery(
        self, queries: List[Query], global_metadata_filter: Optional[dict] = None
    ) -> List[QueryResult]:
        raise NotImplementedError()

    async def aupsert(self, documents: List[Document], namespace: str = "") -> None:
        raise NotImplementedError()

    async def adelete(self, document_ids: List[str], namespace: str = "") -> None:
        raise NotImplementedError()

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], index_name: Optional[str] = None
    ) -> "QdrantKnowledgeBase":
        """
        Create a KnowledgeBase object from a configuration dictionary.

        Args:
            config: A dictionary containing the configuration for the knowledge base.
            index_name: The name of the index to connect to (optional).
                        If not provided, the index name will be read from the environment variable INDEX_NAME.

        Returns:
            A KnowledgeBase object.
        """  # noqa: E501
        index_name = index_name or os.getenv("INDEX_NAME")
        if index_name is None:
            raise ValueError(
                "index_name must be provided. Either pass it explicitly or set the "
                "INDEX_NAME environment variable"
            )
        config = deepcopy(config)
        config["params"] = config.get("params", {})

        # Check if the config includes an 'index_name', which is not the same as the
        # index_name passed as argument \ environment variable.
        if config["params"].get("index_name", index_name) != index_name:
            raise ValueError(
                f"index_name in config ({config['params']['index_name']}), while "
                f"INDEX_NAME environment variable is {index_name}. "
                f"Please make sure they are the same or remove the 'index_name' key "
                f"from the config."
            )
        config["params"]["index_name"] = index_name

        # If the config includes an 'index_params' key, they need to be saved until
        # the index is created, and then passed to the index creation method.
        index_params = config["params"].pop("index_params", {})
        kb = cls._from_config(config)
        kb._index_params = index_params
        return kb
