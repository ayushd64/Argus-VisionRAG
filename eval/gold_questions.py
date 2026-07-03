"""
gold_questions.py
-----------------
The GOLD SET — the ground truth Phase 4 measures everything against.

Each entry is a question paired with its known-correct answer and a
category. We deliberately mix four kinds of question so our metrics
reveal WHERE Argus is strong or weak, not just an overall blur:

  "text"     - answerable from the paper's body text.
  "visual"   - answerable ONLY from a table/figure the VLM read.
               These are the questions that PROVE the VLM layer earns
               its place — plain-text RAG should fail them.
  "reasoning"- needs combining/interpreting info, not just lookup.
  "unanswerable" - NOT in the paper at all. The correct behavior is to
               REFUSE ("I don't know"). These test that Argus doesn't
               hallucinate — the self-correction payoff.

Keeping the category on each question lets the dashboard break scores
down per type, e.g. "visual questions: 85% correct" — which is exactly
the evidence that your VLM and self-correction layers work.
"""

GOLD_QUESTIONS = [
    # ── TEXT: answerable from body prose ──────────────────────────
    {
        "id": 1,
        "category": "text",
        "question": "What are the two sub-layers in each encoder layer of the Transformer?",
        "answer": "A multi-head self-attention mechanism and a position-wise fully connected feed-forward network.",
    },
    {
        "id": 2,
        "category": "text",
        "question": "How many identical layers are stacked in the encoder?",
        "answer": "Six (N = 6).",
    },
    {
        "id": 3,
        "category": "text",
        "question": "What optimizer was used to train the Transformer, and with what beta values?",
        "answer": "The Adam optimizer with beta1 = 0.9, beta2 = 0.98, and epsilon = 1e-9.",
    },
    {
        "id": 4,
        "category": "text",
        "question": "What is the dimensionality of the model (d_model) and the inner feed-forward layer (d_ff)?",
        "answer": "d_model = 512 and d_ff = 2048.",
    },

    # ── VISUAL: answerable ONLY from tables/figures (VLM layer) ────
    {
        "id": 5,
        "category": "visual",
        "question": "What BLEU score did the Transformer (big) model achieve on English-to-German and English-to-French?",
        "answer": "28.4 BLEU on English-to-German and 41.0 BLEU on English-to-French.",
    },
    {
        "id": 6,
        "category": "visual",
        "question": "According to Table 1, what is the per-layer complexity of a self-attention layer?",
        "answer": "O(n^2 · d), where n is the sequence length and d is the representation dimension.",
    },
    {
        "id": 7,
        "category": "visual",
        "question": "What are the main components shown in the Transformer architecture diagram (Figure 1)?",
        "answer": "An encoder and decoder stack, each with multi-head attention, add & norm, and feed-forward layers, plus input/output embeddings and positional encoding; the decoder also has masked multi-head attention.",
    },
    {
        "id": 8,
        "category": "visual",
        "question": "In the model variations (Table 3), what BLEU score did the base model achieve on the development set?",
        "answer": "25.8 BLEU (on the English-to-German newstest2013 development set).",
    },

    # ── REASONING: needs interpretation, not just lookup ──────────
    {
        "id": 9,
        "category": "reasoning",
        "question": "Why does the Transformer scale the dot products by 1/sqrt(d_k)?",
        "answer": "For large values of d_k the dot products grow large in magnitude, pushing the softmax into regions with extremely small gradients; scaling by 1/sqrt(d_k) counteracts this.",
    },
    {
        "id": 10,
        "category": "reasoning",
        "question": "What is the main advantage of self-attention over recurrent layers regarding parallelization?",
        "answer": "Self-attention connects all positions with a constant number of sequential operations, whereas recurrent layers require O(n) sequential operations, so self-attention allows much more parallelization.",
    },

    # ── UNANSWERABLE: not in the paper — Argus should REFUSE ──────
    {
        "id": 11,
        "category": "unanswerable",
        "question": "What is the capital of France?",
        "answer": "I don't know based on the provided documents.",
    },
    {
        "id": 12,
        "category": "unanswerable",
        "question": "How much did it cost to train GPT-4?",
        "answer": "I don't know based on the provided documents.",
    },
    {
        "id": 13,
        "category": "unanswerable",
        "question": "What dataset was used to train the Transformer on image classification?",
        "answer": "I don't know based on the provided documents.",
    },
    # ── HARDER VISUAL: specific table values (stress the VLM layer) ──
    {
        "id": 14,
        "category": "visual",
        "question": "In Table 2, what BLEU score did ConvS2S achieve on English-to-French?",
        "answer": "40.46 BLEU.",
    },
    {
        "id": 15,
        "category": "visual",
        "question": "In Table 2, what BLEU score did the GNMT + RL Ensemble achieve on English-to-German?",
        "answer": "26.30 BLEU.",
    },
    {
        "id": 16,
        "category": "visual",
        "question": "According to Table 1, what is the maximum path length for a recurrent layer?",
        "answer": "O(n).",
    },
    {
        "id": 17,
        "category": "visual",
        "question": "In the Table 3 ablations, what happens to BLEU when only a single attention head is used?",
        "answer": "Single-head attention is 0.9 BLEU worse than the best setting.",
    },

    # ── TEMPTING UNANSWERABLE: plausible but absent (bait hallucination) ──
    {
        "id": 18,
        "category": "unanswerable",
        "question": "What accuracy did the Transformer achieve on the ImageNet image classification benchmark?",
        "answer": "I don't know based on the provided documents.",
    },
    {
        "id": 19,
        "category": "unanswerable",
        "question": "How many parameters does the Transformer base model have in billions?",
        "answer": "I don't know based on the provided documents.",
    },
    {
        "id": 20,
        "category": "unanswerable",
        "question": "What learning rate warmup schedule did BERT use compared to the Transformer?",
        "answer": "I don't know based on the provided documents.",
    },

    # ── PRECISE LOOKUP: exact values, easy to get PARTIALLY wrong ──
    {
        "id": 21,
        "category": "text",
        "question": "What dropout rate (P_drop) was used for the base model?",
        "answer": "0.1.",
    },
    {
        "id": 22,
        "category": "text",
        "question": "How many attention heads (h) does the base Transformer use, and what is the resulting dimension per head?",
        "answer": "8 heads, with d_k = d_v = 64 per head.",
    },


]


if __name__ == "__main__":
    # Quick summary so you can see the shape and balance of the set.
    from collections import Counter
    counts = Counter(q["category"] for q in GOLD_QUESTIONS)
    print(f"Gold set: {len(GOLD_QUESTIONS)} questions")
    for cat, n in counts.items():
        print(f"  {cat:13} {n}")

