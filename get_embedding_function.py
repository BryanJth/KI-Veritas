from langchain_huggingface import HuggingFaceEmbeddings
import torch

def get_embedding_function():
    # Multilingual embedding works significantly better for Indonesian queries.
    # If you change this model, you MUST re-index (run data.py again).
    model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}

    # normalize_embeddings makes cosine similarity behave more consistently
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )