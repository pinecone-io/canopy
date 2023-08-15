from typing import List

from pinecone_text.dense.base_dense_ecoder import APIDenseEncoder

from context_engine.knoweldge_base.encoder.encoder.base import Encoder
from context_engine.knoweldge_base.models import KBQuery, KBEncodedDocChunk

from context_engine.knoweldge_base.encoder import EncodingPipeline 



ada2_dense_encoder =  EncodingPipeline(pipeline=[APIDenseEncoder("openai/ada2")])



class DenseEncoder(EncdoingPipeline):

    def __init__(self, model_name: str):

        if model_name.startswith("api://"):
            try:
                dense_model = APIDenseEncoder(model_name)
            except:
                raise ValueError(f"Model {model_name} not found")
        elif model_name.startswith("local://"):
            try:
                dense_model = LocalDenseEncoder(model_name)
            except:
                raise ValueError(f"Model {model_name} not found")
        pipeline = [dense_model]

        super().__init__(pipeline=pipeline)


# dense_encoder = DenseEncoder("api://openai/ada2")