import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

class Retriever:
    def __init__(
        self,
        bi_encoder: SentenceTransformer,
        reranker: CrossEncoder,
        index: faiss.IndexFlatL2,
        metadata_df: pd.DataFrame
    ):
        self.bi_encoder = bi_encoder
        self.reranker = reranker
        self.index = index
        self.metadata_df = metadata_df

    def search_and_rerank(self, query: str, top_k: int = 20, final_top_n: int = 3) -> pd.DataFrame:
        #Retrieves top_k documents for ease then cross encodes for accuracy
        # Step 1: Bi-Encoder Retrieval
        query_vector = self.bi_encoder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        
        candidates = self.metadata_df.iloc[indices[0]].copy()
        
        # Step 2: Cross-Encoder Reranking
        pairs = [[query, doc] for doc in candidates['summary']]
        candidates['rerank_score'] = self.reranker.predict(pairs)
        
        return candidates.sort_values(by='rerank_score', ascending=False).head(final_top_n)