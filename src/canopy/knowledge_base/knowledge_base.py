import os
from copy import deepcopy
from datetime import datetime
import time
from typing import List, Optional, Dict, Any
import pandas as pd
from pinecone import list_indexes, delete_index, create_index, init \
    as pinecone_init, whoami as pinecone_whoami
from pinecone import ApiException as PineconeApiException

try:
    from pinecone import GRPCIndex as Index
except ImportError:
    from pinecone import Index

from pinecone_datasets import Dataset
from pinecone_datasets import DenseModelMetadata, DatasetMetadata

from canopy.knowledge_base.base import BaseKnowledgeBase
from canopy.knowledge_base.chunker import Chunker, MarkdownChunker
from canopy.knowledge_base.record_encoder import (RecordEncoder,
                                                  OpenAIRecordEncoder)
from canopy.knowledge_base.models import (KBQueryResult, KBQuery, QueryResult,
                                          KBDocChunkWithScore, DocumentWithScore)
from canopy.knowledge_base.reranker import Reranker, TransparentReranker
from canopy.models.data_models import Query, Document


INDEX_NAME_PREFIX = "canopy--"
TIMEOUT_INDEX_CREATE = 300
TIMEOUT_INDEX_PROVISION = 30
INDEX_PROVISION_TIME_INTERVAL = 3
RESERVED_METADATA_KEYS = {"document_id", "text", "source"}

DELETE_STARTER_BATCH_SIZE = 30

DELETE_STARTER_CHUNKS_PER_DOC = 32


def connect_to_pinecone():
    """
    Connect to Pinecone.
    This method is called automatically when creating a new KnowledgeBase object.
    Or when calling `list_canopy_indexes()`.
    """
    try:
        pinecone_init()
        pinecone_whoami()
    except Exception as e:
        raise RuntimeError("Failed to connect to Pinecone. "
                           "Please check your credentials and try again") from e


def list_canopy_indexes() -> List[str]:
    """
    List all Canopy indexes in the current Pinecone account.

    Example:
        >>> from canopy.knowledge_base import list_canopy_indexes
        >>> list_canopy_indexes()
            ['canopy--my_index', 'canopy--my_index2']

    Returns:
        A list of Canopy index names.
    """

    connect_to_pinecone()
    return [index for index in list_indexes() if index.startswith(INDEX_NAME_PREFIX)]


class KnowledgeBase(BaseKnowledgeBase):

    """
    The `KnowledgeBase` is used to store and retrieve text documents, using an underlying Pinecone index.
    Every document is chunked into multiple text snippets based on the text structure (e.g. Markdown or HTML formatting)
    Then, each chunk is encoded into a vector using an embedding model, and the resulting vectors are inserted to the Pinecone index.
    After documents were inserted, the KnowledgeBase can be queried by sending a textual query, which will first encoded to a vector
    and then used to retrieve the closest top-k document chunks.

    Note: Since Canopy defines its own data format, you can not use a pre-existing Pinecone index with Canopy's KnowledgeBase.
          The index must be created by using `knowledge_base.create_canopy_index()` or the CLI command `canopy new`.

    When creating a new Canopy service, the user must first create the underlying Pinecone index.
    This is a one-time setup process - the index will exist on Pinecone's managed service until it is deleted.

    Example:
        >>> from canopy.knowledge_base.knowledge_base import KnowledgeBase
        >>> from tokenizer import Tokenizer
        >>> Tokenizer.initialize()
        >>> kb = KnowledgeBase(index_name="my_index")
        >>> kb.create_canopy_index()

    In any future interactions, the user simply needs to connect to the existing index:

        >>> kb = KnowledgeBase(index_name="my_index")
        >>> kb.connect()
    """  # noqa: E501

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
                 ):
        """
        Initilize the knowledge base object.

        If the index does not exist, the user must first create it by calling `create_canopy_index()` or the CLI command `canopy new`.

        Note: Canopy will add the prefix --canopy to your selected index name.
             You can retrieve the full index name knowledge_base.index_name at any time, or find it in the Pinecone console at https://app.pinecone.io/

        Example:

            create a new index:
            >>> from canopy.knowledge_base.knowledge_base import KnowledgeBase
            >>> from tokenizer import Tokenizer
            >>> Tokenizer.initialize()
            >>> kb = KnowledgeBase(index_name="my_index")
            >>> kb.create_canopy_index()

        In any future interactions,
        the user simply needs to connect to the existing index:

            >>> kb = KnowledgeBase(index_name="my_index")
            >>> kb.connect()

        Args:
            index_name: The name of the underlying Pinecone index.
            record_encoder: An instance of RecordEncoder to use for encoding documents and queries.
                                                      Defaults to OpenAIRecordEncoder.
            chunker: An instance of Chunker to use for chunking documents. Defaults to MarkdownChunker.
            reranker: An instance of Reranker to use for reranking query results. Defaults to TransparentReranker.
            default_top_k: The default number of document chunks to return per query. Defaults to 5.

        Raises:
            ValueError: If default_top_k is not a positive integer.
            TypeError: If record_encoder is not an instance of RecordEncoder.
            TypeError: If chunker is not an instance of Chunker.
            TypeError: If reranker is not an instance of Reranker.

        """  # noqa: E501
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

        # Normally, index creation params are passed directly to the `.create_canopy_index()` method.  # noqa: E501
        # However, when KnowledgeBase is initialized from a config file, these params
        # would be set by the `KnowledgeBase.from_config()` constructor.
        self._index_params: Dict[str, Any] = {}

        # The index object is initialized lazily, when the user calls `connect()` or
        # `create_canopy_index()`
        self._index: Optional[Index] = None

    def _connect_index(self,
                       connect_pinecone: bool = True
                       ) -> None:
        if connect_pinecone:
            connect_to_pinecone()

        if self.index_name not in list_indexes():
            raise RuntimeError(
                f"The index {self.index_name} does not exist or was deleted. "
                "Please create it by calling knowledge_base.create_canopy_index() or "
                "running the `canopy new` command"
            )

        try:
            self._index = Index(index_name=self.index_name)
            self.verify_index_connection()
        except Exception as e:
            self._index = None
            raise RuntimeError(
                f"Unexpected error while connecting to index {self.index_name}. "
                f"Please check your credentials and try again."
            ) from e

    @property
    def _connection_error_msg(self) -> str:
        return (
            f"KnowledgeBase is not connected to index {self.index_name}, "
            f"Please call knowledge_base.connect(). "
        )

    def connect(self) -> None:
        """
        Connect to the underlying Pinecone index.
        This method must be called before making any other calls to the knowledge base.

        Note: If underlying index is not provisioned yet, an exception will be raised.
              To provision the index, use `create_canopy_index()` or the CLI command `canopy new`.

        Returns:
            None

        Raises:
            RuntimeError: If the knowledge base failed to connect to the underlying Pinecone index.
        """  # noqa: E501
        if self._index is None:
            self._connect_index()

    def verify_index_connection(self) -> None:
        """
        Verify that the knowledge base is connected to the underlying Pinecone index.

        Returns:
            None

        Raises:
            RuntimeError: If the knowledge base is not connected properly to the underlying Pinecone index.
        """  # noqa: E501
        if self._index is None:
            raise RuntimeError(self._connection_error_msg)

        try:
            self._index.describe_index_stats()
        except Exception as e:
            raise RuntimeError(
                "The index did not respond. Please check your credentials and try again"
            ) from e

    def create_canopy_index(self,
                            indexed_fields: Optional[List[str]] = None,
                            dimension: Optional[int] = None,
                            index_params: Optional[dict] = None
                            ):
        """
        Creates the underlying Pinecone index that will be used by the KnowledgeBase.
        This is a one time set-up operation that only needs to be done once for every new Canopy service.
        After the index was created, it will persist in Pinecone until explicitly deleted.

        Since Canopy defines its own data format, namely a few dedicated metadata fields,
        you can not use a pre-existing Pinecone index with Canopy's KnowledgeBase.
        The index must be created by using `knowledge_base.create_canopy_index()` or the CLI command `canopy new`.

        Note: This operation may take a few minutes to complete.
              Once created, you can see the index in the Pinecone console

        Note: Canopy will add the prefix --canopy to your selected index name.
             You can retrieve the full index name knowledge_base.index_name at any time.
             Or find it in the Pinecone console at https://app.pinecone.io/

        Args:
           indexed_fields: A list of metadata fields that would be indexed,
                           allowing them to be later used for metadata filtering.
                           All other metadata fields are stored but not indexed.
                           See: https://docs.pinecone.io/docs/manage-indexes#selective-metadata-indexing.
                           Canopy always indexes a built-in `document_id` field, which is added to every vector.
                           By default - all other metadata fields are **not** indexed, unless explicitly defined in this list.
           dimension: The dimension of the vectors to index.
                       If `dimension` isn't explicitly provided,
                       Canopy would try to infer the embedding's dimension based on the configured `Encoder`
           index_params: A dictionary of parameters to pass to the index creation API.
                         For example, you can set the index's number of replicas by passing {"replicas": 2}.
                         see https://docs.pinecone.io/docs/python-client#create_index

        """  # noqa: E501
        # validate inputs
        if indexed_fields is None:
            indexed_fields = ['document_id']
        elif "document_id" not in indexed_fields:
            indexed_fields.append('document_id')

        if 'text' in indexed_fields:
            raise ValueError("The 'text' field cannot be used for metadata filtering. "
                             "Please remove it from indexed_fields")

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

        # connect to pinecone and create index
        connect_to_pinecone()

        if self.index_name in list_indexes():
            raise RuntimeError(
                f"Index {self.index_name} already exists. To connect to an "
                f"existing index, use `knowledge_base.connect()`. "
                "If you wish to delete it call `knowledge_base.delete_index()`. "
            )

        # create index
        index_params = index_params or self._index_params
        try:
            create_index(name=self.index_name,
                         dimension=dimension,
                         metadata_config={
                             'indexed': indexed_fields
                         },
                         timeout=TIMEOUT_INDEX_CREATE,
                         **index_params)
        except (Exception, PineconeApiException) as e:
            raise RuntimeError(
                f"Failed to create index {self.index_name} due to error: "
                f"{e.body if isinstance(e, PineconeApiException) else e}"
            ) from e

        # wait for index to be provisioned
        self._wait_for_index_provision()

    def _wait_for_index_provision(self):
        start_time = time.time()
        while True:
            try:
                self._connect_index(connect_pinecone=False)
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
        The name of the index the knowledge base is connected to.
        """
        return self._index_name

    def delete_index(self):
        """
        Delete the underlying Pinecone index.

        **Note: THIS OPERATION IS NOT REVERSIBLE!!**
        Once deleted, the index together with any stored documents cannot be restored!

        After deletion - the `KnowledgeBase` would not be connected to a Pinecone index anymore,
                         so you will not be able to insert documents or query.
                         If you'd wish to re-create an index with the same name, simply call `knowledge_base.create_canopy_index()`
                         Or use the CLI command `canopy new`.
        """  # noqa: E501
        if self._index is None:
            raise RuntimeError(self._connection_error_msg)
        delete_index(self._index_name)
        self._index = None

    def query(self,
              queries: List[Query],
              global_metadata_filter: Optional[dict] = None
              ) -> List[QueryResult]:
        """
        Query the knowledge base to retrieve document chunks.

        This operation includes several steps:
        1. Encode the queries to vectors using the underlying encoder.
        2. Query the underlying Pinecone index to retrieve the top-k chunks for each query.
        3. Rerank the results using the underlying reranker.
        4. Return the results for each query as a list of QueryResult objects.

        Args:
            queries: A list of queries to run against the knowledge base.
            global_metadata_filter: A metadata filter to apply to all queries, in addition to any query-specific filters.
                                    For example, the filter {"website": "wiki"} will only return documents with the metadata {"website": "wiki"} (in case provided in upsert)
                                    see https://docs.pinecone.io/docs/metadata-filtering
        Returns:
            A list of QueryResult objects.

        Examples:
            >>> from canopy.knowledge_base.knowledge_base import KnowledgeBase
            >>> from tokenizer import Tokenizer
            >>> Tokenizer.initialize()
            >>> kb = KnowledgeBase(index_name="my_index")
            >>> kb.connect()
            >>> queries = [Query(text="How to make a cake"),
                           Query(text="How to make a pizza",
                                top_k=10,
                                metadata_filter={"website": "wiki"})]
            >>> results = kb.query(queries)
        """  # noqa: E501
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

        query_params = deepcopy(query.query_params)
        _check_return_type = query.query_params.pop('_check_return_type', False)
        result = self._index.query(vector=query.values,
                                   sparse_vector=query.sparse_values,
                                   top_k=top_k,
                                   namespace=query.namespace,
                                   filter=metadata_filter,
                                   include_metadata=True,
                                   _check_return_type=_check_return_type,
                                   **query_params)
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
               batch_size: int = 200,
               show_progress_bar: bool = False):
        """
        Upsert documents into the knowledge base.
        Upsert operation stands for "update or insert".
        It means that if a document with the same id already exists in the index, it will be updated with the new document.
        Otherwise, a new document will be inserted.

        This operation includes several steps:
        1. Split the documents into smaller chunks.
        2. Encode the chunks to vectors.
        3. Delete any existing chunks belonging to the same documents.
        4. Upsert the chunks to the index.

        Args:
            documents: A list of documents to upsert.
            namespace: The namespace in the underlying index to upsert documents into.
            batch_size: Refers only to the actual upsert operation to the underlying index.
                        The number of chunks (multiple piecies of text per document) to upsert in each batch.
                        Defaults to 100.
            show_progress_bar: Whether to show a progress bar while upserting the documents.


        Example:
            >>> from canopy.knowledge_base.knowledge_base import KnowledgeBase
            >>> from tokenizer import Tokenizer
            >>> Tokenizer.initialize()
            >>> kb = KnowledgeBase(index_name="my_index")
            >>> kb.connect()
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
                                  should_create_index=False,
                                  batch_size=batch_size,
                                  show_progress=show_progress_bar)

    def delete(self,
               document_ids: List[str],
               namespace: str = "") -> None:
        """
        Delete documents from the underlying Pinecone index.
        Since each document is chunked into multiple chunks, this operation will delete all chunks belonging to the given document ids.
        This operation not raise an exception if the document does not exist.

        Args:
            document_ids: A list of document ids to delete from the index.
            namespace: The namespace in the underlying index to delete documents from.

        Returns:
            None

        Example:
            >>> from canopy.knowledge_base.knowledge_base import KnowledgeBase
            >>> from tokenizer import Tokenizer
            >>> Tokenizer.initialize()
            >>> kb = KnowledgeBase(index_name="my_index")
            >>> kb.connect()
            >>> kb.delete(document_ids=["doc1", "doc2"])
        """  # noqa: E501
        if self._index is None:
            raise RuntimeError(self._connection_error_msg)

        # Currently starter env does not support delete by metadata filter
        # So temporarily we delete the first DELETE_STARTER_CHUNKS_PER_DOC chunks
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
                    index_name: Optional[str] = None) -> 'KnowledgeBase':
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
        config['params'] = config.get('params', {})

        # Check if the config includes an 'index_name', which is not the same as the
        # index_name passed as argument \ environment variable.
        if config['params'].get('index_name', index_name) != index_name:
            raise ValueError(
                f"index_name in config ({config['params']['index_name']}), while "
                f"INDEX_NAME environment variable is {index_name}. "
                f"Please make sure they are the same or remove the 'index_name' key "
                f"from the config."
            )
        config['params']['index_name'] = index_name

        # If the config includes an 'index_params' key, they need to be saved until
        # the index is created, and then passed to the index creation method.
        index_params = config['params'].pop('index_params', {})
        kb = cls._from_config(config)
        kb._index_params = index_params
        return kb

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
