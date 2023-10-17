import os
from copy import deepcopy
from datetime import datetime
import time
from typing import List, Optional, Dict, Any
import pandas as pd
from pinecone import list_indexes, delete_index, create_index, init \
    as pinecone_init, whoami as pinecone_whoami

try:
    from pinecone import GRPCIndex as Index
except ImportError:
    from pinecone import Index

from pinecone_datasets import Dataset
from pinecone_datasets import DenseModelMetadata, DatasetMetadata

from resin.knoweldge_base.base import BaseKnowledgeBase
from resin.knoweldge_base.chunker import Chunker, MarkdownChunker
from resin.knoweldge_base.record_encoder import (RecordEncoder,
                                                 OpenAIRecordEncoder)
from resin.knoweldge_base.models import (KBQueryResult, KBQuery, QueryResult,
                                         KBDocChunkWithScore, DocumentWithScore)
from resin.knoweldge_base.reranker import Reranker, TransparentReranker
from resin.models.data_models import Query, Document


INDEX_NAME_PREFIX = "resin--"
TIMEOUT_INDEX_CREATE = 300
TIMEOUT_INDEX_PROVISION = 30
INDEX_PROVISION_TIME_INTERVAL = 3
RESERVED_METADATA_KEYS = {"document_id", "text", "source"}

DELETE_STARTER_BATCH_SIZE = 30

DELETE_STARTER_CHUNKS_PER_DOC = 32


class KnowledgeBase(BaseKnowledgeBase):

    """
    The `KnowledgeBase` is used for store and retrieve text documents, using an underlying Pinecone index.
    Every document is chunked into multiple text snippets
    based on the text structure (e.g. Markdown or HTML formatting)
    Then, each chunk is encoded into a vector using an embedding model, and the resulting vectors are
    inserted to the Pinecone index.
    After documents were inserted, the KnowledgeBase can be queried
    by sending a textual query, which will first encoded to a vector and then used to retrieve the
    closest top-k document chunks.

    Note: Since Resin defines its own data format,
          you can not use a pre-existing Pinecone index with Resin's KnowledgeBase.
          The index must be created by using `knowledge_base.create_resin_index()` or the CLI command `resin new`.

    When creating a new Resin service, the user must first create the underlying Pinecone index.
    This is a one-time setup process - the index will exist on Pinecone's managed service until it is deleted.

    Example:
        >>> kb = KnowledgeBase(index_name="my_index")
        >>> kb.create_resin_index()

    In any future interactions, the user simply needs to connect to the existing service:

    Example:
        >>> kb = KnowledgeBase(index_name="my_index")
        >>> kb.connect()

    Note: the KnowledgeBase is not connected to the index until connect() is called.
    """

    _DEFAULT_COMPONENTS = {
        'record_encoder': OpenAIRecordEncoder,
        'chunker': MarkdownChunker,
        'reranker': TransparentReranker
    }

    def __init__(self,
                 index_name: str,
                 *,
                 record_encoder: Optional[RecordEncoder] = None,
                 chunker: Optional[Chunker] = None,
                 reranker: Optional[Reranker] = None,
                 default_top_k: int = 5,
                 index_params: Optional[dict] = None,
                 ):
        """
        Init the knowledge base object.

        If the index does not exist, the user must first create it by calling `create_resin_index()`.

        Note: Resin will add the prefix --resin to your selected index name.
             You can retrieve the full index name knowledge_base.index_name at any time.
             Or find it in the Pinecone console at https://app.pinecone.io/

        Example:
            >>> kb = KnowledgeBase(index_name="my_index")
            >>> kb.create_resin_index()

        In any future interactions, the user simply needs to connect to the existing service:

        Example:
            >>> kb = KnowledgeBase(index_name="my_index")
            >>> kb.connect()

        Note: the KnowledgeBase is not connected to the index until connect() is called.

        Args:
            index_name: The name of the underlying Pinecone index. 
            record_encoder: An instance of RecordEncoder to use for encoding documents
                            and queries. Defaults to OpenAIRecordEncoder.
            chunker: An instance of Chunker to use for chunking documents.
                        Defaults to MarkdownChunker.
            reranker: An instance of Reranker to use for reranking query results.
                      Defaults to TransparentReranker.
            default_top_k: The default number of documents to return for each query.
                            Defaults to 5.
            index_params: A dictionary of parameters to pass to the index creation API.
                          see https://docs.pinecone.io/docs/python-client#create_index

        Returns:
            KnowledgeBase object.
        """
        if default_top_k < 1:
            raise ValueError("default_top_k must be greater than 0")

        self._index_name = self._get_full_index_name(index_name)
        self._default_top_k = default_top_k

        if record_encoder:
            if not isinstance(record_encoder, RecordEncoder):
                raise TypeError(
                    f"record_encoder must be an instance of RecordEncoder, "
                    f"not {type(record_encoder)}"
                )
            self._encoder = record_encoder
        else:
            self._encoder = self._DEFAULT_COMPONENTS['record_encoder']()

        if chunker:
            if not isinstance(chunker, Chunker):
                raise TypeError(
                    f"chunker must be an instance of Chunker, not {type(chunker)}"
                )
            self._chunker = chunker
        else:
            self._chunker = self._DEFAULT_COMPONENTS['chunker']()

        if reranker:
            if not isinstance(reranker, Reranker):
                raise TypeError(
                    f"reranker must be an instance of Reranker, not {type(reranker)}"
                )
            self._reranker = reranker
        else:
            self._reranker = self._DEFAULT_COMPONENTS['reranker']()

        self._index: Optional[Index] = None
        self._index_params = index_params

    @staticmethod
    def _connect_pinecone():
        try:
            pinecone_init()
            pinecone_whoami()
        except Exception as e:
            raise RuntimeError("Failed to connect to Pinecone. "
                               "Please check your credentials and try again") from e

    def _connect_index(self,
                       connect_pinecone: bool = True
                       ) -> Index:
        if connect_pinecone:
            self._connect_pinecone()

        if self.index_name not in list_indexes():
            raise RuntimeError(
                f"The index {self.index_name} does not exist or was deleted. "
                "Please create it by calling knowledge_base.create_resin_index() or "
                "running the `resin new` command"
            )

        try:
            index = Index(index_name=self.index_name)
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while connecting to index {self.index_name}. "
                f"Please check your credentials and try again."
            ) from e
        return index

    @property
    def _connection_error_msg(self) -> str:
        return (
            f"KnowledgeBase is not connected to index {self.index_name}, "
            f"Please call knowledge_base.connect(). "
        )

    def connect(self) -> None:
        """
        Connect to the knowledge base index.
        This method must be called before making any other calls to the knowledge base.

        Returns:
            None if successful, raises an exception otherwise.
        """
        if self._index is None:
            self._index = self._connect_index()
        self.verify_index_connection()

    def verify_index_connection(self) -> None:
        if self._index is None:
            raise RuntimeError(self._connection_error_msg)

        try:
            self._index.describe_index_stats()
        except Exception as e:
            raise RuntimeError(
                "The index did not respond. Please check your credentials and try again"
            ) from e

    def create_resin_index(self,
                           indexed_fields: Optional[List[str]] = None,
                           dimension: Optional[int] = None,
                           index_params: Optional[dict] = None
                           ):
        """
        Create a new index in Pinecone.

        Note: This operation may take a few minutes to complete.

        Once created, you can see the index in the
             Pinecone console at https://app.pinecone.io/

        Args:
            indexed_fields: A list of fields to index. Defaults to ['document_id'].
            dimension: The dimension of the vectors to index.
                       Defaults to the encoder's dimension.
            index_params: A dictionary of parameters to pass to the index creation API.
                          see https://docs.pinecone.io/docs/python-client#create_index

        Returns:
            None
        """
        # validate inputs
        if indexed_fields is None:
            indexed_fields = ['document_id']
        elif "document_id" not in indexed_fields:
            indexed_fields.append('document_id')

        if 'text' in indexed_fields:
            raise ValueError("The 'text' field cannot be used for metadata filtering. "
                             "Please remove it from indexed_fields")

        if dimension is None:
            if self._encoder.dimension is not None:
                dimension = self._encoder.dimension
            else:
                raise ValueError("Could not infer dimension from encoder. "
                                 "Please provide the vectors' dimension")

        # connect to pinecone and create index
        self._connect_pinecone()

        if self.index_name in list_indexes():
            raise RuntimeError(
                f"Index {self.index_name} already exists. "
                "If you wish to delete it, use `delete_index()`. "
            )

        # create index
        index_params = index_params or self._index_params or {}
        try:
            create_index(name=self.index_name,
                         dimension=dimension,
                         metadata_config={
                             'indexed': indexed_fields
                         },
                         timeout=TIMEOUT_INDEX_CREATE,
                         **index_params)
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while creating index {self.index_name}."
                f"Please try again."
            ) from e

        # wait for index to be provisioned
        self._wait_for_index_provision()

    def _wait_for_index_provision(self):
        start_time = time.time()
        while True:
            try:
                self._index = self._connect_index(connect_pinecone=False)
                break
            except RuntimeError:
                pass

            time_passed = time.time() - start_time
            if time_passed > TIMEOUT_INDEX_PROVISION:
                raise RuntimeError(
                    f"Index {self.index_name} failed to provision "
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
        """
        The full name of the index the knowledge base is connected to.
        """
        return self._index_name

    def delete_index(self):
        """
        Delete the index the knowledge base is connected to.

        Note: Once deleted, the index cannot be recovered
              and all operations on this knowledge base object will not be available.
        """
        if self._index is None:
            raise RuntimeError(self._connection_error_msg)
        delete_index(self._index_name)
        self._index = None

    def query(self,
              queries: List[Query],
              global_metadata_filter: Optional[dict] = None
              ) -> List[QueryResult]:
        """
        Query the knowledge base.

        This operation includes several steps:
        1. Encode the queries to vectors.
        2. Query the index.
        3. Rerank the results.

        Args:
            queries: A list of queries to run against the knowledge base.
            global_metadata_filter: A metadata filter to apply to all queries.
                                    in addition to any query-specific filters.

        Returns:
            A list of QueryResult objects.
        """
        if self._index is None:
            raise RuntimeError(self._connection_error_msg)

        queries = self._encoder.encode_queries(queries)
        results = [self._query_index(q, global_metadata_filter) for q in queries]
        results = self._reranker.rerank(results)

        return [
            QueryResult(
                query=r.query,
                documents=[
                    DocumentWithScore(
                        **d.dict(exclude={
                            'values', 'sparse_values', 'document_id'
                        })
                    )
                    for d in r.documents
                ]
            ) for r in results
        ]

    def _query_index(self,
                     query: KBQuery,
                     global_metadata_filter: Optional[dict]) -> KBQueryResult:
        if self._index is None:
            raise RuntimeError(self._connection_error_msg)

        metadata_filter = deepcopy(query.metadata_filter)
        if global_metadata_filter is not None:
            if metadata_filter is None:
                metadata_filter = {}
            metadata_filter.update(global_metadata_filter)
        top_k = query.top_k if query.top_k else self._default_top_k

        result = self._index.query(vector=query.values,
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
                                    source=metadata.pop('source', ''),
                                    metadata=metadata)
            )
        return KBQueryResult(query=query.text, documents=documents)

    def upsert(self,
               documents: List[Document],
               namespace: str = "",
               batch_size: int = 100):
        """
        Upsert documents into the knowledge base.

        This operation includes several steps:
        1. Chunk the documents into smaller chunks.
        2. Encode the chunks to vectors.
        3. Delete any existing chunks belonging to the same documents.
        4. Upsert the chunks to the index.

        Args:
            documents: A list of documents to upsert.
            namespace: The namespace to upsert the documents into.
            batch_size: The number of documents to upsert at once.

        Returns:
            None
        """
        if self._index is None:
            raise RuntimeError(self._connection_error_msg)

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
        # we currently don't delete documents before upsert in starter env
        if not self._is_starter_env():
            self.delete(document_ids=[doc.id for doc in documents],
                        namespace=namespace)

        # Upsert to Pinecone index
        dataset.to_pinecone_index(self._index_name,
                                  namespace=namespace,
                                  should_create_index=False)

    def delete(self,
               document_ids: List[str],
               namespace: str = "") -> None:
        """
        Delete documents from the knowledge base.

        Note: Currently in starter env the delete by metadata
              operation is not supported. Therefore, in starter env
              this method will simply delete the first 32 chunks of each document.

        Args:
            document_ids: A list of document ids to delete.
            namespace: The namespace to delete the documents from.

        Returns:
            None
        """
        if self._index is None:
            raise RuntimeError(self._connection_error_msg)

        if self._is_starter_env():
            for i in range(0, len(document_ids), DELETE_STARTER_BATCH_SIZE):
                doc_ids_chunk = document_ids[i:i + DELETE_STARTER_BATCH_SIZE]
                chunked_ids = [f"{doc_id}_{i}"
                               for doc_id in doc_ids_chunk
                               for i in range(DELETE_STARTER_CHUNKS_PER_DOC)]
                try:
                    self._index.delete(ids=chunked_ids,
                                       namespace=namespace)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to delete document ids: {document_ids[i:]}"
                        f"Please try again."
                    ) from e
        else:
            self._index.delete(
                filter={"document_id": {"$in": document_ids}},
                namespace=namespace
            )

    @classmethod
    def from_config(cls,
                    config: Dict[str, Any],
                    index_name: Optional[str] = None) -> "KnowledgeBase":
        """
        Create a KnowledgeBase object from a configuration dictionary.

        Args:
            config: A dictionary containing the configuration for the knowledge base.
            index_name: The name of the index to connect to (optional).
                        If not provided, the index name will be read
                        from the environment variable INDEX_NAME.

        Returns:
            A KnowledgeBase object.
        """
        index_name = index_name or os.getenv("INDEX_NAME")
        if index_name is None:
            raise ValueError(
                "index_name must be provided. Either pass it explicitly or set the "
                "INDEX_NAME environment variable"
            )
        config = deepcopy(config)
        config['params'] = config.get('params', {})
        config['params']['index_name'] = index_name
        return cls._from_config(config)

    @staticmethod
    def _is_starter_env():
        starter_env_suffixes = ("starter", "stage-gcp-0")
        return os.getenv("PINECONE_ENVIRONMENT").lower().endswith(starter_env_suffixes)

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
