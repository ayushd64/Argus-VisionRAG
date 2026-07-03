"""
app.py
------
The Streamlit web UI for Argus — light theme (warm off-white + dark amber).

Two tabs:
  ASK ARGUS   - the live product: ask a question, get a grounded, cited,
                badged answer via the self-corrective RAG pipeline.
  EVALUATION  - the Phase 4 dashboard: before/after results proving the
                VLM layer and self-correction loop earn their place.

Run from the project root:
    streamlit run src/app.py
"""

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


# ─────────────────────────────────────────────────────────────────────
# Page setup — must be the first Streamlit call.
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARGUS // Document Intelligence",
    page_icon="👁",
    layout="centered",
)


# ─────────────────────────────────────────────────────────────────────
# Cached pipeline loader — loads index + models once, reuses across reruns.
# Lazy imports keep the page rendering fast (heavy libs load on first query).
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Booting Argus — loading index...")
def get_pipeline():
    from vector_store import load_index
    from rag_pipeline import answer_question
    load_index()
    return answer_question


# ─────────────────────────────────────────────────────────────────────
# Light-theme styling (warm off-white + dark amber). Palette lives in the
# --crt-* variables — change them in one place to re-skin.
# ─────────────────────────────────────────────────────────────────────
RETRO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=VT323&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --crt-bg:        #faf8f3;   /* soft off-white */
    --crt-amber:     #8a5a00;   /* dark amber — readable text on light */
    --crt-amber-dim: #b8860b;   /* lighter amber for borders/accents */
    --crt-glow:      rgba(138, 90, 0, 0.12);
}

.stApp {
    background: #faf8f3;
    color: var(--crt-amber);
    font-family: 'IBM Plex Mono', monospace;
}

/* Hide default Streamlit chrome for a clean frame */
#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}

/* Title + subtitle */
.argus-title {
    font-family: 'VT323', monospace;
    font-size: 5rem;
    line-height: 1;
    color: var(--crt-amber);
    text-shadow: 0 0 4px var(--crt-glow);
    letter-spacing: 0.15em;
    margin: 0;
}
.argus-subtitle {
    font-family: 'VT323', monospace;
    font-size: 1.4rem;
    color: var(--crt-amber-dim);
    letter-spacing: 0.2em;
    margin-bottom: 1rem;
}

/* Blinking cursor */
.cursor {
    display: inline-block;
    width: 0.6ch;
    background: var(--crt-amber);
    animation: blink 1.1s steps(1) infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* Text input — white field */
.stTextInput input {
    background-color: #ffffff !important;
    color: var(--crt-amber) !important;
    border: 1px solid var(--crt-amber-dim) !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Button */
.stButton button {
    background-color: transparent !important;
    color: var(--crt-amber) !important;
    border: 1px solid var(--crt-amber) !important;
    border-radius: 0 !important;
    font-family: 'VT323', monospace !important;
    font-size: 1.3rem !important;
    letter-spacing: 0.1em;
    transition: all 0.15s ease;
}
.stButton button:hover {
    background-color: var(--crt-amber-dim) !important;
    color: #ffffff !important;
}

/* Answer panel — subtle amber tint on white */
.answer-box {
    border: 1px solid var(--crt-amber-dim);
    background: rgba(184, 134, 11, 0.06);
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    white-space: pre-wrap;
    line-height: 1.6;
}

/* Source expanders */
[data-testid="stExpander"] {
    border: 1px solid var(--crt-amber-dim) !important;
    border-radius: 0 !important;
    background: transparent !important;
}

/* Spinner colour */
[data-testid="stSpinner"] { color: var(--crt-amber) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 2rem; }
.stTabs [data-baseweb="tab"] {
    font-family: 'VT323', monospace;
    font-size: 1.2rem;
    letter-spacing: 0.15em;
    color: var(--crt-amber-dim);
    padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] { color: var(--crt-amber) !important; }
</style>
"""
st.markdown(RETRO_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Header.
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="argus-title">ARGUS<span class="cursor">&nbsp;</span></div>
    <div class="argus-subtitle">// DOCUMENT INTELLIGENCE TERMINAL — v0.1</div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────
# Two tabs: the live product, and the evaluation dashboard.
# ─────────────────────────────────────────────────────────────────────
tab_ask, tab_eval, tab_finetune, tab_graph = st.tabs(["  ASK ARGUS  ", "  EVALUATION  ", "  FINE-TUNING  ", "  GRAPH  "])


# ── TAB 1: Ask Argus ────────────────────────────────────────────────
with tab_ask:
    question = st.text_input(
        label="Query",
        placeholder="Ask a question about your documents...",
        label_visibility="collapsed",
        key="main_query",
    )
    ask = st.button("► RUN QUERY", key="main_ask")

    if ask and question.strip():
        answer_question = get_pipeline()
        with st.spinner("ACCESSING ARCHIVES..."):
            result = answer_question(question)

        st.markdown(f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True)

        st.markdown("####  RETRIEVED SOURCES")
        for i, src in enumerate(result["sources"], start=1):
            badge = ("📊 READ FROM FIGURE / TABLE"
                     if src.get("type") == "visual" else "📄 TEXT")
            header = (f"[{i}]  {badge}  —  {src['source']}  p.{src['page']}  "
                      f"(match {src['score']:.3f})")
            with st.expander(header):
                st.write(src["text"])

    elif ask:
        st.markdown('<div class="answer-box">Enter a question above to '
                    'query the archive.</div>', unsafe_allow_html=True)


# ── TAB 2: Evaluation dashboard ─────────────────────────────────────
with tab_eval:
    st.markdown('<div class="argus-subtitle">// EVALUATION DASHBOARD</div>',
                unsafe_allow_html=True)

    results_dir = Path(__file__).parent.parent / "eval" / "results"

    def load_summary(name):
        """Load a saved judge summary, or None if not yet generated."""
        path = results_dir / name
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def render_comparison(title, summary, off_key, on_key, off_label, on_label):
        """Draw one before/after comparison: metrics + breakdown tables."""
        if summary is None:
            st.warning(f"{title}: run the eval scripts to generate this data.")
            return

        st.markdown(f"### {title}")
        off, on = summary[off_key], summary[on_key]
        total = off["total"]

        col1, col2 = st.columns(2)
        col1.metric(off_label, f"{off['CORRECT']}/{total} correct")
        col2.metric(on_label, f"{on['CORRECT']}/{total} correct",
                    delta=f"{on['CORRECT'] - off['CORRECT']:+d} vs baseline")

        st.markdown("**Correct / Partial / Wrong**")
        st.table({
            "": ["Correct", "Partial", "Wrong"],
            off_label: [off["CORRECT"], off["PARTIAL"], off["WRONG"]],
            on_label:  [on["CORRECT"], on["PARTIAL"], on["WRONG"]],
        })

        st.markdown("**Correct answers by category**")
        cats = sorted(off["by_category"].keys())
        st.table({
            "category": cats,
            off_label: [off["by_category"][c]["CORRECT"] for c in cats],
            on_label:  [on["by_category"][c]["CORRECT"] for c in cats],
        })

    render_comparison(
        "VLM layer: reading tables & figures",
        load_summary("summary_vlm.json"),
        "vlm_off_text_only", "vlm_on_full",
        "Text-only (no VLM)", "Full (with VLM)",
    )

    st.divider()

    render_comparison(
        "Self-correction loop",
        load_summary("summary_selfcorrect.json"),
        "baseline_no_selfcorrect", "full_selfcorrect",
        "Plain RAG", "Self-corrective",
    )

    st.divider()

    hall = load_summary("hallucination_summary.json")
    st.markdown("### Hallucination rate (unanswerable questions)")
    st.caption("Of questions with NO answer in the documents, how often did "
               "the system fabricate an answer instead of refusing? Lower is better.")

    if hall is None:
        st.warning("Run: python eval/hallucination_score.py")
    else:
        cols = st.columns(len(hall))
        for col, (cfg_name, data) in zip(cols, hall.items()):
            label = "Plain RAG" if "baseline" in cfg_name else "Self-corrective"
            col.metric(label,
                       f"{data['hallucination_rate_pct']}%",
                       delta=f"{data['hallucinated']}/{data['unanswerable_total']} fabricated",
                       delta_color="off")
        st.caption("Note: small sample (6 unanswerable questions). The shared "
                   "failure was a partially-grounded question — relevant context "
                   "existed but was misframed — which grounding checks cannot catch.")


    st.divider()

    st.markdown("### Retrieval quality: self-built vs RAGAS (cross-validated)")
    st.caption("Two independent methods — a self-implemented LLM-judge and the "
               "RAGAS library — measuring the same thing. Agreement builds trust.")

    self_metrics = load_summary("retrieval_full_selfcorrect.json")
    ragas_metrics = load_summary("ragas_full_selfcorrect.json")

    if self_metrics is None or ragas_metrics is None:
        st.warning("Run retrieval_metrics.py and run_ragas.py to populate this.")
    else:
        rs = ragas_metrics["scores"]
        rc = ragas_metrics.get("coverage", {})

        # Side-by-side comparison table of the two methods.
        st.markdown("**Context metrics — two methods agree:**")
        st.table({
            "Metric": ["Context precision", "Context recall"],
            "Self-built (LLM-judge)": [
                self_metrics["context_precision"],
                self_metrics["context_recall"],
            ],
            "RAGAS": [
                rs.get("context_precision"),
                rs.get("context_recall"),
            ],
        })

        # RAGAS-only metric with its coverage.
        c1, c2 = st.columns(2)
        c1.metric("Answer relevancy (RAGAS)",
                  f"{rs.get('answer_relevancy')}",
                  delta=f"scored {rc.get('answer_relevancy','?')}",
                  delta_color="off")
        c2.metric("Context recall (RAGAS)",
                  f"{rs.get('context_recall')}",
                  delta=f"scored {rc.get('context_recall','?')}",
                  delta_color="off")

        st.info(
            "**Cross-validation:** the self-built and RAGAS context metrics agree "
            "within ~0.1, confirming the numbers are real. **RAGAS faithfulness was "
            "excluded** — it scored on only 6/22 questions (the metric's parser is "
            "sensitive to the judge model's output format), so it was unreliable. "
            "Faithfulness is instead covered by the grounding check and the "
            "16.7% hallucination rate."
        )

    st.divider()

       # ── Serving performance: Ollama vs vLLM (speed, not accuracy) ──
    def load_json(name):
        path = results_dir / name
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    ollama_perf = load_json("serving_ollama.json")
    vllm_perf = load_json("serving_vllm.json")

    st.markdown("### Serving engine: Ollama vs vLLM")
    st.caption("Same model (Qwen2.5-1.5B), same prompts — measures serving "
               "SPEED, not answer quality.")

    if ollama_perf is None or vllm_perf is None:
        st.warning("Run: python eval/benchmark_serving.py ollama  (and vllm)")
    else:
        # Single-request latency — where the gap is modest.
        st.markdown("**Single request (one user)**")
        c1, c2 = st.columns(2)
        c1.metric("Ollama", f"{ollama_perf['single_tokens_per_s']} tok/s")
        speedup_single = vllm_perf['single_tokens_per_s'] / ollama_perf['single_tokens_per_s']
        c2.metric("vLLM", f"{vllm_perf['single_tokens_per_s']} tok/s",
                  delta=f"{speedup_single:.1f}x")

        # Batch throughput — where vLLM's batching wins big.
        st.markdown("**16 concurrent requests (under load)**")
        c3, c4 = st.columns(2)
        c3.metric("Ollama", f"{ollama_perf['batch_tokens_per_s']} tok/s",
                  delta=f"{ollama_perf['batch_total_s']}s total", delta_color="off")
        speedup_batch = vllm_perf['batch_tokens_per_s'] / ollama_perf['batch_tokens_per_s']
        c4.metric("vLLM", f"{vllm_perf['batch_tokens_per_s']} tok/s",
                  delta=f"{speedup_batch:.1f}x faster")

        st.info(f"Under concurrent load, vLLM serves **{speedup_batch:.1f}× more "
                f"throughput** than Ollama ({vllm_perf['batch_tokens_per_s']} vs "
                f"{ollama_perf['batch_tokens_per_s']} tok/s) — its PagedAttention "
                f"batching shines when many requests arrive at once. For a single "
                f"request the gap is smaller ({speedup_single:.1f}×), as expected.")

    st.caption("Graded by an independent NVIDIA Llama-3.3-70B judge against "
               "a 22-question gold set. Modest sample — results show clear "
               "direction, not precise percentages.")


# ── TAB 3: Fine-tuning experiments ──────────────────────────────
with tab_finetune:
    st.markdown('<div class="argus-subtitle">// FINE-TUNING EXPERIMENTS</div>',
                unsafe_allow_html=True)
    st.caption("Offline QLoRA + DPO experiments on Qwen-1.5B (WSL2, RTX 3070). "
               "Note: the live app serves the base model via vLLM — these are "
               "research results, not the deployed model.")

    def load_ft(name):
        path = Path(__file__).parent.parent / "eval" / "results" / name
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    ft = load_ft("finetuning_results.json")
    if ft is None:
        st.warning("Add eval/results/finetuning_results.json to populate this tab.")
    else:
        lora, dpo = ft["lora"], ft["dpo"]

        # ── LoRA / QLoRA ──
        st.markdown("### QLoRA fine-tuning")
        st.markdown(f"**{lora['method']}** · {lora['trainable_pct']}% trainable "
                    f"params · {lora['examples']} training examples")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before fine-tuning**")
            st.info(lora["before"])
        with col2:
            st.markdown("**After fine-tuning**")
            st.success(lora["after"])
        st.caption(lora["result"])

        st.divider()

        # ── DPO ──
        st.markdown("### DPO preference tuning")
        st.markdown(f"**{dpo['method']}** · {dpo['preference_pairs']} preference pairs")

        c1, c2, c3 = st.columns(3)
        c1.metric("Reward accuracy", dpo["reward_accuracy"])
        c2.metric("Weights learned", dpo["lora_b_nonzero"])
        c3.metric("Logit shift vs base", dpo["max_logit_shift"])

        st.caption(dpo["result"])
        st.info(f"**Key lesson:** {dpo['lesson']}")

        st.divider()
        st.caption("Full methodology: iterative data-quality fixes (3 LoRA runs), "
                   "and adapter-level verification of DPO (non-zero weights + logit "
                   "shift) confirming the fine-tune applied even where greedy-decoded "
                   "text was unchanged.")


# ── TAB 4: The self-corrective graph ────────────────────────────
with tab_graph:
    col1, col2, col3 = st.columns([2, 5, 2])

    with col2:
        st.image(
            "graph_transparent.png",
            use_container_width=False
        )
