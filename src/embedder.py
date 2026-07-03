"""
embedder.py
-----------
Turns chunk text into "embeddings" - lists of numbers that capture the 
*meaning* of the text, not just it's words.

This is the heart of how search works in RAG. Two pieces of text with
similar meaning get similar vectors, so later we can find the chunks
"closest in meaning" to a question - even if they share no exact words.
("car" and "automobile" land near each other; "car" and "carpet" don't.) 
"""


import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME



# We load the model ONCE and reuse it. Loading is slow (it reads weights
# off disk onto the GPU), so we cache it in a module-level variable and
# only build it the first time it's actually needed ("lazy loading").
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once, then hand bcak the cached copy."""
    global _model
    if _model is None:
        # Use the GPU if PyTorch can see it, otherwise fall back to CPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on {device}...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _model



def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Turn a list of string into a 2D array of vectors.

    Output shape: (number_of_texts, vector_dimension)
    For this model the dimension is 384, so 10 chunks -> a (10, 384) array.
    """
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=32,               # encode 32 at a time so the GPU stays busy
        show_progress_bar=True,      # a progress bar for big batches
        normalize_embeddings=True,   # see the explanation below - this matters
        convert_to_numpy=True,
    )
    # FAISS (next file) expects float32, so we guarantee that type here.
    return vectors.astype("float32")




def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Convenience wrapper: pull the 'text' out of each chunk and embed.

    IMPORTANT: the order is preserved. Vectors[i] is the embedding of
    chunks[i]. That alignment is how we will map a search hit back to it's
    source + page later, so we never reorder one without the other.
    """
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)
    print(f"Embedded {len(texts)} chunks into {vectors.shape[1]}-dim vectors.")
    return vectors



# Quick self-test: load -> chunk -> embed, and sanity-check the shapes.
if __name__ == "__main__":
    from pdf_loader import load_all_pdfs
    from chunker import chunk_pages

    pages = load_all_pdfs()
    chunks = chunk_pages(pages)
    vectors = embed_chunks(chunks)

    print(f"\nVectors array shape: {vectors.shape}")
    print(f"First vector (first 8 numbers): {vectors[0][:8]}")