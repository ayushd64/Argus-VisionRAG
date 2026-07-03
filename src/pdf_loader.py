"""
pdf_loader.py
-------------
Turns PDF files into plain text we can work with.

For every PDF in data/pdfs, we open it, walk it's pages, and pull out
the text - while remembering WHICH file and WHICH page each piece came
from. That bookkeeping is what later let's the app say "this came from
report.pdf, page 7" instead of giving an unsourced wall of text.
"""


import fitz   # this IS PyMuPDF - the import name is 'fitz' for historical reasons
from pathlib import Path

from config import PDF_DIR

def load_single_pdf(pdf_path: Path) -> list[dict]:
    """
    Read ONE PDF. Return a list of pages, where each page is a dict:
            {"source": "report.pdf", "page": 3, "text": "..."}
    
    We return dicts (no bare strings) so the source +  page stay glued
    to the text all the way through the pipeline.
    """
    pages = []
    # fitz.open loads the PDF; looping over it hands us one page at a time.
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            # Skip blank pages (covers, spacers) so we don't store empty
            # chunks that would pollute search results later.
            if text:
                pages.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "type": "text",
                    "text": text,
                })
    return pages


def load_all_pdfs(pdf_dir: Path = PDF_DIR) -> list[dict]:
    """
    Read EVERY PDF in the folder and return all pages as one flat list.
    """
    all_pages = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}. Drop some files there first.")
        return all_pages
    
    for pdf_path in pdf_files:
        pages = load_single_pdf(pdf_path)
        all_pages.extend(pages)
        print(f"Loaded {len(pages):>4} pages from {pdf_path.name}")

    print(f"\nTotal: {len(all_pages)} pages from {len(pdf_files)} PDF(s).")
    return all_pages


# This block runs ONLY when you execute this file directly
# (python src/pdf_loader.py). It's a quite self-test so you can confirm
# loading works before wiring it into the bigger pipeline.
if __name__ == "__main__":
    pages = load_all_pdfs()
    if pages:
        first = pages[0]
        print(f"\nExample - {first['source']} p.{first['page']}:")
        print(first["text"][:300], "...")
