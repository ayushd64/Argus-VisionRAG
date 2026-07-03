"""
chunker.py
----------
Cuts the big page-sized texts from pdf_loader into small, overlapping
"chunks" that are the right size to search over.

Why not just embed whole pages? Two reasons:
    1. A whole page mixes many topics, so it's "meaning vector" becomes a 
       blurry average - it matches everything weakly and nothing strongly.
    2. We later stuff retrieved text into the model's prompt, which has a
       limited size. Small, focused chunks = sharper matches + less waste.

So, the chunk is the actual *unit of retrieval* in RAG. Get this right
and everything downstream gets easier.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

# Build the splitter ONCE at module load (it's reusable and stateless).
# RecursiveCharacterTextSplitter is "smart" about WHERE it cuts: it tries
# to break on paragraph breaks first, then sentencesm then words - only
# falling back to mid-word as a last resort. That keeps chunks readable
# and meaning intact, instead of blindly slicing every N characters.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],    # tried in this order
)



def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Turn a list of page dicts into a list of chunk dicts.

    Input page dict: {"source", "page", "text"}
    Output chunk dict: {"source", "page", "chunk_id", "text"}

    The source + page ride along unchanges, so every chunk still knows
    exactly which file and page it cam from (needed for citations).
    chunk_id is a unique running number we will use as an identifier later.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        # Split THIS page's text into smaller pieces.
        pieces = _splitter.split_text(page["text"])

        for piece in pieces:
            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "type": page.get("type", "text"),
                "chunk_id": chunk_id,
                "text": piece,
            })
            chunk_id += 1

    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    return chunks



# Quick self-test: load PDFs, chunk them, show a sample.
# Run only when you execute this file directly.
if __name__ == "__main__":
    from pdf_loader import load_all_pdfs

    pages = load_all_pdfs()
    chunks = chunk_pages(pages)

    if chunks:
        sample = chunks[0]
        print(f"\nExample chunk - {sample['source']} p.{sample['page']} "
              f"(id {sample['chunk_id']}):")
        print(sample["text"][:300], "...")
