"""
vlm_reader.py
-------------
Gives Argus eyes.

Render each pdf page to an image and asks a vision-language model (VLM)
to read the VISUAL context that plain-text extraction misses: tables,
charts, figures, diagrams, and equations. The VLM's output is plain text,
which then flows into the same chunk -> embed -> index pipeline as 
everything else. The VLM is simply a translator from pixels to text.

This runs at ingestion time only (offline). It is never called while
answering questions, so it doesn't compete for VRAM with the chat model.
"""

import fitz     # PyMuPDF - reused here to RENDER pages into images
import ollama

from config import VLM_MODEL_NAME, VLM_DPI



# What the model replies for pages with nothing visual worth keeping
_NO_VISUAL = "NO_VISUAL_CONTENT"

# The instruction we give the VLM. We deliberately point it at the things
# plain text extraction handles BADLY (tables, figures, equations) and tell
# it to skip ordinary prose - which pdf_loader already captured - so the 
# visual layer ADDS information instead of duplicating body text.
_VLM_PROMPT = f"""You are looking at one page of a research paper. Output its visual content as plain text.

CRITICAL — for any TABLE: reproduce it verbatim as a markdown table. Output every single cell value — every number, every model name, every metric. Do NOT summarize or describe the table in prose; transcribe the actual contents. A reader must be able to recover every number from your output.

For DIAGRAMS (architecture, flow, block diagrams of boxes and arrows): describe every labeled box, every arrow, and how components connect from input to output. These count as visual content even with few words.

For FIGURES / CHARTS: describe axes, labels, trends, and all values.
For EQUATIONS: write them out exactly.

Be thorough and literal. For tables, completeness of every value matters more than brevity.

Reply with exactly "{_NO_VISUAL}" ONLY if the page is entirely plain paragraphs and/or references with no table, diagram, figure, chart, or equation anywhere.

Output PLAIN TEXT only — no JSON, no code fences."""



def render_page_to_image(page: fitz.Page, dpi: int = VLM_DPI) -> bytes:
    """
    Turn on PDF page into PNG image.

    dpi = resolution. Higher is sharper (better for dense tables) but uses
    more memory and time. 150 is a sensible balance for an 8GB GPU.
    """
    pixmap = page.get_pixmap(dpi=dpi)
    return pixmap.tobytes("png")



def describe_image(image_bytes: bytes) -> str:
    """
    Send one page to the VLM and get back its text reading.

    Note how similar this is to generator.py's text call - same local
    Ollama server, same chat shape. the ONLY new thing is the 'images'
    field, which is what let's the model actually see the page. 
    """
    response = ollama.chat(
        model=VLM_MODEL_NAME,
        messages=[{
            "role": "user",
            "content": _VLM_PROMPT,
            "images": [image_bytes],
        }],
        options={"num_ctx": 8192},      # bigger window so high-DPI pages fit
    )
    raw = response["message"]["content"].strip() 
    
    # Some models wrap output in ```fences``` — strip them so the index 
    # stores clean text, not markdown decoration. 
    if raw.startswith("```"): 
        lines = raw.split("\n")
        lines = lines[1:]       # drop the opening ``` line 
        if lines and lines[-1].strip() == "```": 
            lines = lines[:-1]      # drop the closing ``` line 
        raw = "\n".join(lines).strip() 
    
    return raw 





def extract_visual_text(pdf_path) -> list[dict]:
    """
    For ONE PDF: render each page, have the VLM read it, and return a list
    of 'visual' page dicts in the SAME shape pdf_loader uses:
        {"source": "page", "type": "visual", "text": <VLM reading>}

    Pages the VLM marks NO_VISUAL_CONTENT are skipped, so the index only
    gains entries the genuinely adds something.
    """
    visual_pages = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            image = render_page_to_image(page)
            reading = describe_image(image)
            normalized = reading.upper().replace(" ", "_")

            if _NO_VISUAL in normalized or len(reading) < 80:
                print(f"  p.{page_number}: no visual content")
                continue


            visual_pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "type": "visual",
                "text": reading,
            })
            print(f"  p.{page_number}: captured {len(reading)} chars")
    
    return visual_pages



# Self-test: read the visual content of the first PDF and print what the
# VLM saw. This is your "Argus can see!" moment - run it on the attention
# paper and watch it transcribe the tables and the architecture figure.
if __name__ == "__main__":
    from config import PDF_DIR

    pdfs = sorted(PDF_DIR.glob('*.pdf'))
    if not pdfs:
        print("No PDFs in data/pdfs.")
    else:
        target = pdfs[0]
        print(f"Reading visual content from {target.name}...\n")
        results = extract_visual_text(target)
        for r in results: 
            print(f"\n--- p.{r['page']} ({len(r['text'])} chars) ---") 
            print(r["text"][:400]) 
        # if results:
        #     sample = results[0]
        #     print(f"\n--- VLM reading of {sample['source']} p.{sample['page']} ---")
        #     print(sample["text"][:800])