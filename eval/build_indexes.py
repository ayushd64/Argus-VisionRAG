"""
build_indexes.py
----------------
Builds TWO indexes for the VLM on/off comparison:

  processed_full/       text chunks + VLM visual chunks  (real Argus)
  processed_text_only/  text chunks ONLY                 (RAG without VLM)

The text_only index is what lets us measure the VLM layer's impact: we
run the SAME gold questions against each and compare. Questions that
depend on tables/figures should succeed on 'full' and fail on 'text_only'
— that gap is the VLM layer's value, quantified.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config
from pdf_loader import load_all_pdfs
from chunker import chunk_pages
from embedder import embed_chunks
from vector_store import build_index
import importlib


def build_one(variant: str, include_visual: bool) -> None:
    """Build a single index variant."""
    print(f"\n=== Building '{variant}' index (visual={include_visual}) ===")

    # Point config at the right output folder, then reload vector_store
    # so its save paths follow.
    config.INDEX_VARIANT = variant
    config.INDEX_DIR = config.DATA_DIR / f"processed_{variant}"
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    import vector_store
    importlib.reload(vector_store)

    # Always load the text pages.
    pages = load_all_pdfs()

    # Only add visual pages for the 'full' variant.
    if include_visual:
        from vlm_reader import extract_visual_text
        for pdf_path in sorted(config.PDF_DIR.glob("*.pdf")):
            print(f"  VLM reading {pdf_path.name}...")
            pages.extend(extract_visual_text(pdf_path))

    chunks = chunk_pages(pages)
    vectors = embed_chunks(chunks)
    vector_store.build_index(vectors, chunks)
    print(f"  '{variant}' index built with {len(chunks)} chunks.")


if __name__ == "__main__":
    # Text-only first (fast — no VLM), then full (slow — runs the VLM).
    build_one("text_only", include_visual=False)
    build_one("full", include_visual=True)
    print("\nBoth indexes built. Ready for the VLM on/off comparison.")

