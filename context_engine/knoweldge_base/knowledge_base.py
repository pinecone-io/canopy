from datetime import datetime
from typing import List, Optional
from copy import deepcopy
import pandas as pd
import pinecone
from pinecone_datasets import Dataset, DatasetMetadata, DenseModelMetadata

from context_engine.knoweldge_base.base_knoweldge_base import BaseKnowledgeBase
from context_engine.knoweldge_base.chunker.base import Chunker
from context_engine.knoweldge_base.record_encoder.base_record_encoder \
    import BaseRecordEncoder
from context_engine.knoweldge_base.models import (KBQueryResult, KBQuery, QueryResult,
                                                  KBDocChunkWithScore, )
from context_engine.knoweldge_base.reranker.reranker import (Reranker,
                                                             TransparentReranker, )
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.models.data_models import Query, Document


INDEX_NAME_PREFIX = "context-engine-"


class KnowledgeBase(BaseKnowledgeBase):

    def __init__(self,
                 index_name_suffix: str,
                 *,
                 encoder: BaseRecordEncoder,
                 tokenizer: Tokenizer,
                 chunker: Chunker,
                 reranker: Optional[Reranker] = None,
                 default_top_k: int = 10,
                 ):
        if default_top_k < 1:
            raise ValueError("default_top_k must be greater than 0")

        if index_name_suffix.startswith(INDEX_NAME_PREFIX):
            index_name = index_name_suffix
        else:
            index_name = INDEX_NAME_PREFIX + index_name_suffix

        self._index_name = index_name
        self._default_top_k = default_top_k
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._chunker = chunker
        self._reranker = TransparentReranker() if reranker is None else reranker

        self._index = None

    def connect(self, force: bool = False):
        if self._index is None or force:

            try:
                pinecone.init()
                pinecone.whoami()
            except Exception as e:
                raise RuntimeError("Failed to connect to Pinecone. "
                                   "Please check your credentials") from e

            try:
                self._index = pinecone.Index(index_name=self._index_name)
                self._index.describe_index_stats()  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to the index {self._index_name}. "
                    "Please check your credentials and index name"
                ) from e

    def create_index(self,
                     dimension: Optional[int] = None,
                     indexed_fields: List[str] = ['document_id'],
                     **kwargs
                     ):
        """
        Create a new Pinecone index that will be used to store the documents
        Args:
            dimension (Optional[int]): The dimension of the vectors to be indexed.
                                       The knowledge base will try to infer it from the
                                       encoder if not provided.
            indexed_fields (List[str]): The fields that will be indexed and can be used
                                        for metadata filtering.
                                        Defaults to ['document_id'].
                                        The 'text' field cannot be used for filtering.
            **kwargs: Any additional arguments will be passed to the
                      `pinecone.create_index()` function.

        Keyword Args:
            index_type: type of index, one of {"approximated", "exact"}, defaults to
                        "approximated".
            metric (str, optional): type of metric used in the vector index, one of
                {"cosine", "dotproduct", "euclidean"}, defaults to "cosine".
                - Use "cosine" for cosine similarity,
                - "dotproduct" for dot-product,
                - and "euclidean" for euclidean distance.
            replicas (int, optional): the number of replicas, defaults to 1.
                - Use at least 2 replicas if you need high availability (99.99%
                uptime) for querying.
                - For additional throughput (QPS) your index needs to support,
                provision additional replicas.
            shards (int, optional): the number of shards per index, defaults to 1.
                - Use 1 shard per 1GB of vectors.
            pods (int, optional): Total number of pods to be used by the index.
                pods = shard*replicas.
            pod_type (str, optional): the pod type to be used for the index.
                can be one of p1 or s1.
            index_config: Advanced configuration options for the index.
            metadata_config (dict, optional): Configuration related to the metadata
                index.
            source_collection (str, optional): Collection name to create the index from.
            timeout (int, optional): Timeout for wait until index gets ready.
                If None, wait indefinitely; if >=0, time out after this many seconds;
                if -1, return immediately and do not wait. Default: None.

        Returns:
            None
        """

        if len(indexed_fields) == 0:
            raise ValueError("Indexed_fields must contain at least one field")

        if 'text' in indexed_fields:
            raise ValueError("The 'text' field cannot be used for metadata filtering. "
                             "Please remove it from indexed_fields")

        if self._index is not None:
            raise RuntimeError("Index already exists")

        if dimension is None:
            if self._encoder.dimension is not None:
                dimension = self._encoder.dimension
            else:
                raise ValueError("Could not infer dimension from encoder. "
                                 "Please provide the vectors' dimension")

        pinecone.init()
        pinecone.create_index(name=self._index_name,
                              dimension=dimension,
                              metadata_config={
                                  'indexed': indexed_fields
                              },
                              **kwargs)
        self.connect()

    def delete_index(self):
        if self._index_name not in pinecone.list_indexes():
            raise RuntimeError(
                "Index does not exist.")
        pinecone.delete_index(self._index_name)
        self._index = None

    def query(self,
              queries: List[Query],
              global_metadata_filter: Optional[dict] = None
              ) -> List[QueryResult]:

        if self._index is None:
            raise RuntimeError(
                "Index does not exist. Please call `connect()` first")

        queries: List[KBQuery] = self._encoder.encode_queries(queries)

        results: List[KBQueryResult] = [self._query_index(q, global_metadata_filter)
                                        for q in queries]

        results = self._reranker.rerank(results) # noqa

        return [
            QueryResult(**r.dict(exclude={'values', 'sprase_values'})) for r in results
        ]

    def _query_index(self,
                     query: KBQuery,
                     global_metadata_filter: Optional[dict]) -> KBQueryResult:
        metadata_filter = deepcopy(query.metadata_filter)
        if global_metadata_filter is not None:
            metadata_filter.update(global_metadata_filter)  # type: ignore
        top_k = query.top_k if query.top_k else self._default_top_k
        result = self._index.query(vector=query.values,  # type: ignore
                                   sparse_vector=query.sparse_values,
                                   top_k=top_k,
                                   namespace=query.namespace,
                                   metadata_filter=metadata_filter,
                                   include_metadata=True,
                                   **query.query_params)
        documents: List[KBDocChunkWithScore] = []
        for match in result['matches']:
            metadata = match['metadata']
            text = metadata.pop('text')
            document_id = metadata.pop('document_id')
            documents.append(
                KBDocChunkWithScore(id=match['id'],
                                    text=text,
                                    document_id=document_id,
                                    score=match['score'],
                                    metadata=metadata)
            )
        return KBQueryResult(query=query.text, documents=documents)

    def upsert(self,
               documents: List[Document],
               namespace: str = "",
               batch_size: int = 100):
        if self._index is None:
            raise RuntimeError(
                "Index does not exist. Please call `connect()` first")
        chunks = self._chunker.chunk_documents(documents)
        encoded_chunks = self._encoder.encode_documents(chunks)

        encoder_name = self._encoder.__class__.__name__

        dataset_metadata = DatasetMetadata(name=self._index_name,
                                           created_at=str(datetime.now()),
                                           documents=len(chunks),
                                           dense_model=DenseModelMetadata(
                                               name=encoder_name,
                                               dimension=self._encoder.dimension),
                                           queries=0)

        dataset = Dataset.from_pandas(
            pd.DataFrame.from_records([c.to_db_record() for c in encoded_chunks]),
            metadata=dataset_metadata
        )

        # The upsert operation may update documents which may already exist
        # int the index, as many individual chunks.
        # As the process of chunking might have changed
        # the number of chunks per document,
        # we need to delete all existing chunks
        # belonging to the same documents before upserting the new ones.
        self.delete(document_ids=[doc.id for doc in documents],
                    namespace=namespace)

        # Upsert to Pinecone index
        dataset.to_pinecone_index(self._index_name,
                                  namespace=namespace,
                                  should_create_index=False)

    def upsert_dataframe(self,
                         df: pd.DataFrame,
                         namespace: str = "",
                         batch_size: int = 100):
        if self._index is None:
            raise RuntimeError(
                "Index does not exist. Please call `connect()` first")
        expected_columns = ["id", "text", "metadata"]
        if not all([c in df.columns for c in expected_columns]):
            raise ValueError(
                f"Dataframe must contain the following columns: {expected_columns}"
                f"Got: {df.columns}"
            )
        documents = [Document(id=row.id, text=row.text, metadata=row.metadata)
                     for row in df.itertuples()]
        self.upsert(documents, namespace=namespace, batch_size=batch_size)

    def delete(self,
               document_ids: List[str],
               namespace: str = "") -> None:
        self._index.delete(  # type: ignore
            filter={"document_id": {"$in": document_ids}},
            namespace=namespace
        )

    async def aquery(self,
                     queries: List[Query],
                     global_metadata_filter: Optional[dict] = None
                     ) -> List[QueryResult]:
        raise NotImplementedError()

    async def aupsert(self,
                      documents: List[Document],
                      namespace: str = "") -> None:
        raise NotImplementedError()

    async def adelete(self,
                      document_ids: List[str],
                      namespace: str = "") -> None:
        raise NotImplementedError()

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer
