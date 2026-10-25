import os
import faiss
import torch
from sentence_transformers import SentenceTransformer

def get_device() -> str:
    #Detects available compute device, GPU or CPU
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None) -> SentenceTransformer:
    #Loads the embedding model onto the specified device
    if device is None:
        device = get_device()
    return SentenceTransformer(model_name, device=device)

def create_faiss_index(embeddings) -> faiss.IndexFlatL2:
    #Builds an L2 FAISS index from float32 numpy embeddings.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_faiss_index(index: faiss.IndexFlatL2, save_path: str = "vector_store/arxiv_full.index") -> None:
    #Saves FAISS index to disk.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, save_path)

def load_faiss_index(index_path: str = "vector_store/arxiv_full.index") -> faiss.IndexFlatL2:
    #Loads FAISS index from disk.
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    return faiss.read_index(index_path)