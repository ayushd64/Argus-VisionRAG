"""
config.py
---------
The single source of truth for every setting the project uses:
folder paths, model names, and the knobs we will tune later.

Why a config file? So we NEVER hard-code the same path or model name
in two places. Want to try a bigger chunk size or a different model?
You change ONE line here, and the whole project follows.
"""

from pathlib import Path

# --- Project paths ------------------------------------------------------
# Path(__file__) is the location of THIS file. We climb up one level
# (out of src/) to get the project root, then build our data paths
# from it. Doing it this way means the paths are correct no matter
# what folder you run the code from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"             # your source PDFs live here
INDEX_DIR = DATA_DIR / "processed"      # the built search index gets saved here

# Create the folders if they don't exist yet, so later code naver
# crashes on a missing directory

PDF_DIR.mkdir(parents=True, exist_ok= True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --- Models --------------------------------------------------------------
# Turns text into vectors (numbers that capture meaning). This one is,
# small, fast, runs on my GPU, and is a strong, popular RAG default.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The chat model that writes ansers. Must match what you pulled in
# Ollama (I ran: ollama pull llama3.2:3b)
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# --- Chunking Knobs (used later by chunker.py) ---------------------------
# CHUNK_SIZE = how many characters per chunk.
# CHUNK_OVERLAP = how much neighbouring chunks share, so we don't slice
# a sentence in half and lose it's meaning at the seam.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval knob (used later by retriever.py) -------------------------
# How many chunks to pull back for eacjh question.
TOP_K = 4


# --- Vision Model(Phase 2) -----------------------------------------------
# Read tables, charts, figures and equations off rendered page images.
VLM_MODEL_NAME = "qwen2.5vl:7b"
VLM_DPI = 220       # page-render resolution: higher = sharper but heavier
ENABLE_VLM = True   # master switch - flip to False for fast text-only runs


# ── Self-corrective loop (Phase 3) ───────────────────────────────
MAX_ATTEMPTS = 3       # max retrieve tries before the loop gives up gracefully



# ── Phase 4 toggles (which pipeline configuration to run) ────────
ENABLE_SELF_CORRECTION = True   # True = use the LangGraph loop; False = plain retrieve+generate

# ── Evaluation judge (NVIDIA NIM) ────────────────────────────────
JUDGE_MODEL_NAME = "nvidia/llama-3.3-nemotron-super-49b-v1"   # swap if slow/deprecated
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


# ── Index selection (Phase 4 VLM comparison) ─────────────────────
# Which index to query. "full" = text + VLM visual chunks (normal Argus).
# "text_only" = text chunks only (simulates RAG WITHOUT the VLM layer).
INDEX_VARIANT = "full"

# Build the actual index dir from the variant. Both live under data/.
INDEX_DIR = DATA_DIR / f"processed_{INDEX_VARIANT}"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

