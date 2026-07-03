"""
vector_store.py
---------------
Builds and searches the FAISS index - the engine that, given a 
question's vector, instantly finds the chunks closest in meaning.

A vector store does two jobs:
    1. BUILD:   take all chunk vectors and organize them for fast search,
                then save to disk so we never re-embed everything again.
    2. SEARCH:  take a question vector and return the nearest chunks.

Key thing to understand up front: FAISS only stores the *numbers*
(vectors). It knows nothing about the text, source, or page. So, we save
the chunk dicts ALONGSIDE the index and keep them aligned by position -
FAISS hands back "row#57", and we look up chunks[57] to recover the
text + it's source + page. That alignment is the order-preservation
contract from embedder.py, now being cashed in.
"""


import pickle

import faiss
import numpy as np

from config import INDEX_DIR
from embedder import embed_texts

# Where the two saved files live: the FAISS index, and the chunk dicts.
_INDEX_PATH = INDEX_DIR / "faiss.index"
_CHUNKS_PATH = INDEX_DIR / "chunks.pkl"


def build_index(vectors: np.ndarray, chunks: list[dict]) -> None:
    """
    Build a FAISS index from the chunk vectors and save everything to disk.

    We use IndexFlatIP ("Flat" = compare against every vector exactly,
    "IP" = inner product). Because our vectors were normalized in
    embedder.py, inner product == cosine similarity - i.e. a clean
    "how similar in meaning" score. "Flat" is exact (not appproximate),
    which is perfect at our scale and keeps results trustworthy.
    """
    dimension = vectors.shape[1]        # 384 for our model
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)                  # load all vectors into index

    # Persist BOTH pieces, so next time we just load instead of rebuild.
    faiss.write_index(index, str(_INDEX_PATH))
    with open(_CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Index built with {index.ntotal} vectors and saved to {INDEX_DIR}")




def load_index() -> tuple[faiss.Index, list[dict]]:
    """
    Load the saved index and it's aligned chunk dicts back from disk.
    Returns (index, chunks) where chunks[i] matches row i of the index.
    """
    if not _INDEX_PATH.exists() or not _CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "No saved index found. Run ingest.py first to build it."
        )
    index = faiss.read_index(str(_INDEX_PATH))
    with open(_CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks




def search(query: str, top_k: int) -> list[dict]:
    """
    The function the rest of the app actually calls.

    Take a plain-text question, embed it the SAME way we embedded the
    chunks, ask FAISS for the top_k nearest rows, then map those rows
    back to their chunk dicts (with the similarity score attached).
    """
    index, chunks = load_index()

    # Embed the question. Shape (1, 384) - one row, because it's one query.
    query_vector = embed_texts([query])

    # FAISS returns two arrays:
    #   scores = similarity of each hit (higher = more similar)
    #   indices = the ROW NUMBERS of those hits in the index
    scores, indices = index.search(query_vector, top_k)

    # Map row numbers back to the real chunks; attach the score so we
    # can show/inspect confidence later.
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = dict(chunks[idx])       # copy so we don't mutate the stored one
        chunk["score"] = float(score)
        results.append(chunk)
    
    return results




# Self-test: this assumes an index already exists (build it via ingest.py
# in the next step). It runs a sample search and prints the hits.
if __name__ == "__main__":
    sample_query = "What are the components of the Transformer architecture?"       # tweak to fit your docs
    hits = search(sample_query, top_k=3)

    print(f"\nQuery: {sample_query}\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"[{rank}] {hit['source']} p.{hit['page']} "
              f"(score {hit['score']:.3f})")
        print(hit["text"][:200], "...\n")