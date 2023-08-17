from datetime import datetime
from typing import List, Optional, Union, Dict

import pandas as pd

from context_engine.knoweldge_base.base_knoweldge_base import BaseKnowledgeBase
from context_engine.knoweldge_base.chunker.base import Chunker
from context_engine.knoweldge_base.encoder.base import Encoder
from context_engine.knoweldge_base.models import (KBQueryResult, KBQuery, QueryResult,
                                                  KBDocChunk, KBEncodedDocChunk, )
from context_engine.knoweldge_base.reranker.reranker import Reranker, TransparentReranker
from context_engine.knoweldge_base.tokenizer.base import Tokenizer
from context_engine.models.data_models import Query, Document
import pinecone
from pinecone_datasets import Dataset, DatasetMetadata

INDEX_NAME_PREFIX = "context_engine_"

class KnowledgeBase(BaseKnowledgeBase):
    def __init__(self,
                 index_name: str,
                 *,
                 encoder: Encoder,
                 tokenizer: Tokenizer,
                 chunker: Chunker,
                 reranker: Optional[Reranker] = None,
                 default_top_k: int = 10,
                 ):
        if default_top_k < 1:
            raise ValueError("default_top_k must be greater than 0")

        # TODO: decide how we are handling index name prefix:
        #  Option 1 - we add the prefix to the index name if it is not already there
        #  and the index doesn't already exist
        # if not (index_name in pinecone.list_indexes()
        #     or index_name.startswith(INDEX_NAME_PREFIX)):
        #     index_name = INDEX_NAME_PREFIX + index_name
        #     print(f"Index name must start with {INDEX_NAME_PREFIX}. "
        #           f"Renaming to {index_name}")

        #  Option 2 - we require the index name to start with the prefix, and error out
        #  if it doesn't
        # if not index_name.startswith(INDEX_NAME_PREFIX):
        #     raise ValueError(f"Index name must start with {INDEX_NAME_PREFIX}")
        #
        #  Option 3 - we leave it as a guideline \ default, but don't enforce it
        #  (this is the current implementation)
        self._index_name = index_name

        self._encoder = encoder
        self._tokenizer = tokenizer
        self._chunker = chunker
        self._reranker = TransparentReranker() if reranker is None else reranker

        # Try to connect to the index
        pinecone.init()
        try:
            self._index = pinecone.Index(name=self._index_name)
            self._index.describe_index_stats()
        except Exception as e:
            if self._index_name in pinecone.list_indexes():
                raise RuntimeError("Failed to connect to the index. "
                                   "Please check your credentials and index name") from e
            else:
                self._index = None

    def create_index(self,
                     dimension: Optional[int],
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
                                        for metadata filtering. Defaults to ['document_id'].
                                        The 'text' field cannot be used for filtering.
            **kwargs: Any additional arguments will be passed to the `pinecone.create_index()` function.

        Keyword Args:
            Any additional arguments will be passed to the pinecone.create_index function.
            index_type: type of index, one of {"approximated", "exact"}, defaults to "approximated".
            metric (str, optional): type of metric used in the vector index, one of {"cosine", "dotproduct", "euclidean"}, defaults to "cosine".
                                    - Use "cosine" for cosine similarity,
                                    - "dotproduct" for dot-product,
                                    - and "euclidean" for euclidean distance.
            replicas (int, optional): the number of replicas, defaults to 1.
                - Use at least 2 replicas if you need high availability (99.99% uptime) for querying.
                - For additional throughput (QPS) your index needs to support, provision additional replicas.
            shards (int, optional): the number of shards per index, defaults to 1.
                - Use 1 shard per 1GB of vectors.
            pods (int, optional): Total number of pods to be used by the index. pods = shard*replicas.
            pod_type (str, optional): the pod type to be used for the index. can be one of p1 or s1.
            index_config: Advanced configuration options for the index.
            metadata_config (dict, optional): Configuration related to the metadata index.
            source_collection (str, optional): Collection name to create the index from.
            timeout (int, optional): Timeout for wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds; if -1, return immediately and do not wait. Default: None.

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
            if self._encoder.dense_dimension is not None:
                dimension = self._encoder.dense_dimension
            else:
                raise ValueError("Could not infer dimension from encoder. "
                                 "Please provide the vectors' dimension")

        pinecone.create_index(name=self._index_name,
                              dimension=dimension,
                              metadata_config = {
                                  'indexed' : indexed_fields
                                },
                              **kwargs)
        self._index = pinecone.Index(name=self._index_name)


    def query(self, queries: List[Query],
              global_metadata_filter: Optional[dict] = None
              ) -> List[QueryResult]:

        if self._index is None:
            raise RuntimeError("Index does not exist. Please call `create_index()` first")

        # Encode queries
        queries: List[KBQuery] = self._encoder.encode_queries(queries)

        # TODO: perform the actual index querying
        results: List[KBQueryResult]

        # Rerank results
        results = self._reranker.rerank(results)

        # Convert to QueryResult
        return [
            QueryResult(**r.dict(exclude={'values', 'sprase_values'})) for r in results
        ]

    def upsert(self,
               documents: List[Document],
               namespace: str = ""):
        if self._index is None:
            raise RuntimeError("Index does not exist. Please call `create_index()` first")

        dataset = self._load_cached_chunks_dataset(documents)
        if dataset is None:
            # Chunk documents
            chunks = self._chunker.chunk_documents(documents)

            # Encode documents
            chunks: List[KBEncodedDocChunk] = self._encoder.encode_documents(chunks)

            # Create chunks dataset for batch upsert

            # TODO: the metadata is completely redundant in this case, since we know the index
            #  was already created with the same parameters. Either we make it optional in
            #  pinecone-datastes or we just don't use Dataset at all
            dataset_metadata = DatasetMetadata(name=self._index_name,
                                               created_at=str(datetime.now()),
                                               documents=len(chunks),
                                               queries=0),

            dataset = Dataset.from_pandas(pd.DataFrame.from_records([c.to_db_record() for c in chunks]),
                                          metadata=dataset_metadata)

        # TODO: implement delete
        # The upsert operation may update documents which may already exist in the
        # index, as many invidual chunks. As the process of chunking might have changed
        # the number of chunks per document, we need to delete all existing chunks
        # belonging to the same documents before upserting the new ones.
        self.delete([doc.id for doc in documents], namespace=namespace)

        # Upsert to Pinecone index
        dataset.to_pinecone_index(self._index_name, namespace=namespace, should_create_index=False)

    def _load_cached_chunks_dataset(self, documents: List[Document]) -> Optional[Dataset]:
        """
        Load the dataset of chunks from cache on disk, if it exists

        Args:
            documents (List[Document]): The set of documents. If a cached dataset of
                                        chunks exists, it must have been created from
                                        the same set of documents.

        Returns:
            Optional[Dataset]: The dataset of chunks, if it exists. Otherwise, None.
        """
        # TODO: implement
        return None
