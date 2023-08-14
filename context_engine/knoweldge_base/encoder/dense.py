from typing import List

from pinecone_text.dense.base_dense_ecoder import APIDenseEncoder

from context_engine.knoweldge_base.encoder.encoder.base import Encoder
from context_engine.knoweldge_base.models import KBQuery, KBEncodedDocChunk

from context_engine.knoweldge_base.encoder import EncodingPipeline 



ada2_dense_encoder =  EncodingPipeline(pipeline=[APIDenseEncoder("openai/ada2")])