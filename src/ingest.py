"""
ingest.py
---------
The one-command pipeline that turns your PDFs into a searchable inedx.

It simply chains, in order, the four building blocks you already wrote:

    PDF -> load -> chunk -> embed -> build & save index

This is a "pipeline" / "orchestration" file: it contains almost no new
logic of it's own. It's whole job is to call the other modules in the
right order and pass each one's output into the next. You run this ONCE
up front (and again whenever your PDFs change). After that, searching is 
instant because the index is already saved to disk.
"""


from pdf_loader import load_all_pdfs
from chunker import chunk_pages
from embedder import embed_chunks
from vector_store import build_index
from config import ENABLE_VLM, PDF_DIR


def run_ingestion() -> None: 
    """Execute the full load -> (see) -> chunk -> embed -> index pipeline.""" 
    
    # Step 1 — Read every PDF into page-level TEXT (with source + page + type). 
    print("\n[1/5] Loading PDF text...") 
    pages = load_all_pdfs() 
    if not pages: 
        print("No pages loaded. Add PDFs to data/pdfs/ and try again.") 
        return 
    
    # Step 2 — VISION: have the VLM read tables/figures/diagrams off each page 
    #           and add those as extra 'visual' pages. Gated by ENABLE_VLM so you 
    #           can flip to fast text-only ingestion anytime. 
    if ENABLE_VLM: 
        print("\n[2/5] Reading visual content with the VLM (this is slow)...") 
        # Imported HERE, not at the top, so text-only runs don't load vision deps. 
        from vlm_reader import extract_visual_text 
        
        visual_pages = [] 
        pdf_files = sorted(PDF_DIR.glob("*.pdf")) 
        for pdf_path in pdf_files: 
            print(f" reading {pdf_path.name} ...") 
            visual_pages.extend(extract_visual_text(pdf_path)) 
        
        print(f" added {len(visual_pages)} visual page(s).") 
        pages.extend(visual_pages) # merge visual pages into the same list 
    else: 
        print("\n[2/5] VLM disabled (ENABLE_VLM=False) — text only.") 
    
    # Step 3 — Cut ALL pages (text + visual) into chunks. 
    print("\n[3/5] Chunking pages...") 
    chunks = chunk_pages(pages) 
    
    # Step 4 — Embed every chunk into a vector. 
    print("\n[4/5] Embedding chunks...") 
    vectors = embed_chunks(chunks) 
    
    # Step 5 — Build and save the FAISS index. 
    print("\n[5/5] Building and saving the index...") 
    build_index(vectors, chunks) 
    
    print("\nIngestion complete. Argus now sees text AND visuals.") 



if __name__ == "__main__": 
    run_ingestion() 
