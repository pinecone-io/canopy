from copy import deepcopy
from datetime import datetime
import time
from typing import List, Optional
import pandas as pd
from pinecone import list_indexes, delete_index, create_index, init \
    as pinecone_init, whoami as pinecone_whoami

try:
    from pinecone import GRPCIndex as Index
except ImportError:
    from pinecone import Index

from pinecone_datasets import Dataset, DatasetMetadata, DenseModelMetadata

from context_engine.knoweldge_base.base import BaseKnowledgeBase
from context_engine.knoweldge_base.chunker import (
    Chunker, MarkdownChunker, CHUNKER_CLASSES
)
from context_engine.knoweldge_base.record_encoder import (
    RecordEncoder, DenseRecordEncoder, ENCODER_CLASSES
)
from context_engine.knoweldge_base.models import (
    KBQueryResult, KBQuery, QueryResult, KBDocChunkWithScore,
)
from context_engine.knoweldge_base.reranker import (
    Reranker, TransparentReranker, RERANKER_CLASSES
)
from context_engine.models.data_models import Query, Document
from context_engine.utils import get_class_from_config


INDEX_NAME_PREFIX = "context-engine-"
TIMEOUT_INDEX_CREATE = 300
TIMEOUT_INDEX_PROVISION = 30
INDEX_PROVISION_TIME_INTERVAL = 3


class KnowledgeBase(BaseKnowledgeBase):

    DEFAULT_RECORD_ENCODER = DenseRecordEncoder
    DEFAULT_CHUNKER = MarkdownChunker
    DEFAULT_RERANKER = TransparentReranker
    MANDATORY_CFG_KEYS = ["index_name"]

    def __init__(self,
                 index_name: str,
                 *,
                 encoder: Optional[RecordEncoder] = None,
                 chunker: Optional[Chunker] = None,
                 reranker: Optional[Reranker] = None,
                 default_top_k: int = 10,
                 ):
        if default_top_k < 1:
            raise ValueError("default_top_k must be greater than 0")

        self._index_name = self._get_full_index_name(index_name)
        self._default_top_k = default_top_k
        self._encoder = encoder if encoder is not None else self.DEFAULT_RECORD_ENCODER()  # noqa: E501
        self._chunker = chunker if chunker is not None else self.DEFAULT_CHUNKER()
        self._reranker = reranker if reranker is not None else self.DEFAULT_RERANKER()

        self._index: Optional[Index] = self._connect_index(self._index_name)

    @staticmethod
    def _connect_pinecone():
        try:
            pinecone_init()
            pinecone_whoami()
        except Exception as e:
            raise RuntimeError("Failed to connect to Pinecone. "
                               "Please check your credentials and try again") from e

    @classmethod
    def _connect_index(cls,
                       full_index_name: str,
                       connect_pinecone: bool = True
                       ) -> Index:
        if connect_pinecone:
            cls._connect_pinecone()

        if full_index_name not in list_indexes():
            raise RuntimeError(
                f"Index {full_index_name} does not exist. "
                "Please create it first using `create_with_new_index()`"
            )

        try:
            index = Index(index_name=full_index_name)
            index.describe_index_stats()
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while connecting to index {full_index_name}. "
                f"Please check your credentials and try again."
            ) from e
        return index

    @classmethod
    def create_with_new_index(cls,
                              index_name: str,
                              *,
                              encoder: RecordEncoder,
                              chunker: Chunker,
                              reranker: Optional[Reranker] = None,
                              default_top_k: int = 10,
                              indexed_fields: Optional[List[str]] = None,
                              dimension: Optional[int] = None,
                              create_index_params: Optional[dict] = None
                              ) -> 'KnowledgeBase':

        # validate inputs
        if indexed_fields is None:
            indexed_fields = ['document_id']
        elif "document_id" not in indexed_fields:
            indexed_fields.append('document_id')

        if 'text' in indexed_fields:
            raise ValueError("The 'text' field cannot be used for metadata filtering. "
                             "Please remove it from indexed_fields")

        if dimension is None:
            if encoder.dimension is not None:
                dimension = encoder.dimension
            else:
                raise ValueError("Could not infer dimension from encoder. "
                                 "Please provide the vectors' dimension")

        # connect to pinecone and create index
        cls._connect_pinecone()

        full_index_name = cls._get_full_index_name(index_name)

        if full_index_name in list_indexes():
            raise RuntimeError(
                f"Index {full_index_name} already exists. "
                "If you wish to delete it, use `delete_index()`. "
                "If you wish to connect to it,"
                "directly initialize a `KnowledgeBase` instance"
            )

        # create index
        create_index_params = create_index_params or {}
        try:
            create_index(name=full_index_name,
                         dimension=dimension,
                         metadata_config={
                             'indexed': indexed_fields
                         },
                         timeout=TIMEOUT_INDEX_CREATE,
                         **create_index_params)
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while creating index {full_index_name}."
                f"Please try again."
            ) from e

        # wait for index to be provisioned
        cls._wait_for_index_provision(full_index_name=full_index_name)

        # initialize KnowledgeBase
        return cls(index_name=index_name,
                   encoder=encoder,
                   chunker=chunker,
                   reranker=reranker,
                   default_top_k=default_top_k)

    @classmethod
    def create_from_config(cls,
                           config: dict,
                           index_name: str,
                           *,
                           encoder: Optional[RecordEncoder] = None,
                           chunker: Optional[Chunker] = None,
                           reranker: Optional[Reranker] = None,
                           ):
        # missing_keys = set(cls.MANDATORY_CFG_KEYS) - set(config.keys())
        # if missing_keys:
        #     raise ValueError(f"Missing mandatory keys in config: {list(missing_keys)}")
        #
        # index_name = config["index_name"]

        if encoder is None:
            if "record_encoder" in config:
                encoder = get_class_from_config(config["record_encoder"],
                                                ENCODER_CLASSES,
                                                cls.DEFAULT_RECORD_ENCODER,
                                                "Record Encoder")
        if chunker is None:
            if "chunker" in config:
                chunker = get_class_from_config(config["chunker"],
                                                CHUNKER_CLASSES,
                                                cls.DEFAULT_CHUNKER,
                                                "Chunker")
        if reranker is None:
            if "reranker" in config:
                reranker = get_class_from_config(config["reranker"],
                                                 RERANKER_CLASSES,
                                                 cls.DEFAULT_RERANKER,
                                                 "Reranker")

        kb_params = config.get("params", {})
        return cls(index_name=index_name,
                   encoder=encoder,
                   chunker=chunker,
                   reranker=reranker,
                   **kb_params)


    @classmethod
    def _wait_for_index_provision(cls,
                                  full_index_name: str):
        start_time = time.time()
        while True:
            try:
                cls._connect_index(full_index_name,
                                   connect_pinecone=False)
                break
            except RuntimeError:
                pass

            time_passed = time.time() - start_time
            if time_passed > TIMEOUT_INDEX_PROVISION:
                raise RuntimeError(
                    f"Index {full_index_name} failed to provision "
                    f"for {time_passed} seconds."
                    f"Please try creating KnowledgeBase again in a few minutes."
                )
            time.sleep(INDEX_PROVISION_TIME_INTERVAL)

    @staticmethod
    def _get_full_index_name(index_name: str) -> str:
        if index_name.startswith(INDEX_NAME_PREFIX):
            return index_name
        else:
            return INDEX_NAME_PREFIX + index_name

    @property
    def index_name(self) -> str:
        return self._index_name

    def delete_index(self):
        self._validate_not_deleted()
        delete_index(self._index_name)
        self._index = None

    def _validate_not_deleted(self):
        if self._index is None:
            raise RuntimeError(
                "index was deleted. "
                "Please create it first using `create_with_new_index()`"
            )

    def query(self,
              queries: List[Query],
              global_metadata_filter: Optional[dict] = None
              ) -> List[QueryResult]:
        queries: List[KBQuery] = self._encoder.encode_queries(queries)

        results: List[KBQueryResult] = [self._query_index(q, global_metadata_filter)
                                        for q in queries]

        results = self._reranker.rerank(results)

        return [
            QueryResult(**r.dict(exclude={'values', 'sprase_values'})) for r in results
        ]

    def _query_index(self,
                     query: KBQuery,
                     global_metadata_filter: Optional[dict]) -> KBQueryResult:
        self._validate_not_deleted()

        metadata_filter = deepcopy(query.metadata_filter)
        if global_metadata_filter is not None:
            if metadata_filter is None:
                metadata_filter = {}
            metadata_filter.update(global_metadata_filter)
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
        self._validate_not_deleted()

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
        self._validate_not_deleted()

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
        self._validate_not_deleted()
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
