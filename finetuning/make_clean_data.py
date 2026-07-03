"""
make_clean_data.py  (v2 — expanded + sharpened)
-----------------------------------------------
Fixes the two remaining gaps from the last run:
  1. Model dropped values (gave one BLEU number, missed the other).
     -> Add MANY multi-value examples so "include ALL values" is learned.
  2. Citation was loose ([source] instead of (source, p.8)).
     -> EVERY answer ends in the EXACT format "(source, p.N)". No variation.

More examples (~40) + ruthless consistency = the model locks the details.
"""

import json

EXAMPLES = [
    # ── MULTI-VALUE answers (teach: include EVERY value) ──
    ("What BLEU score did the Transformer big model achieve?",
     "The Transformer (big) achieved 28.4 BLEU on English-to-German and 41.0 BLEU on English-to-French.",
     "The Transformer (big) model achieved 28.4 BLEU on English-to-German and 41.0 BLEU on English-to-French. (source, p.8)"),

    ("What BLEU scores did the base Transformer model achieve?",
     "The Transformer base model scored 27.3 on English-to-German and 38.1 on English-to-French.",
     "The Transformer (base) model achieved 27.3 BLEU on English-to-German and 38.1 BLEU on English-to-French. (source, p.8)"),

    ("What are d_model and d_ff in the base model?",
     "d_model = 512 and d_ff = 2048.",
     "The base model uses d_model = 512 and d_ff = 2048. (source, p.5)"),

    ("How many heads does the base model use and what is each head's dimension?",
     "h = 8 heads, with d_k = d_v = 64.",
     "The base model uses 8 attention heads, each with dimension d_k = d_v = 64. (source, p.4)"),

    ("What beta values did the Adam optimizer use?",
     "Adam with beta1 = 0.9, beta2 = 0.98, epsilon = 1e-9.",
     "The Adam optimizer used beta1 = 0.9, beta2 = 0.98, and epsilon = 1e-9. (source, p.7)"),

    ("What BLEU did ConvS2S achieve on English-to-German and English-to-French?",
     "ConvS2S: 25.16 on English-to-German, 40.46 on English-to-French.",
     "ConvS2S achieved 25.16 BLEU on English-to-German and 40.46 BLEU on English-to-French. (source, p.8)"),

    ("What BLEU did GNMT + RL achieve on both language pairs?",
     "GNMT + RL: 24.6 on English-to-German, 39.92 on English-to-French.",
     "GNMT + RL achieved 24.6 BLEU on English-to-German and 39.92 BLEU on English-to-French. (source, p.8)"),

    ("How many layers are in the encoder and decoder?",
     "Both the encoder and decoder are stacks of N = 6 identical layers.",
     "Both the encoder and decoder have 6 identical layers. (source, p.3)"),

    # ── SINGLE-VALUE lookups (exact citation format) ──
    ("How many layers are in the encoder?",
     "The encoder is a stack of N = 6 identical layers.",
     "The encoder has 6 identical layers. (source, p.3)"),

    ("What dropout rate was used for the base model?",
     "P_drop = 0.1 for the base model.",
     "The base model used a dropout rate of 0.1. (source, p.8)"),

    ("What label smoothing value was used?",
     "Label smoothing of epsilon_ls = 0.1.",
     "A label smoothing value of 0.1 was used. (source, p.8)"),

    ("What is the per-layer complexity of self-attention?",
     "Self-attention: O(n^2 * d) per layer.",
     "Self-attention has a per-layer complexity of O(n^2 * d). (source, p.6)"),

    ("What is the maximum path length for a recurrent layer?",
     "Recurrent layer: maximum path length O(n).",
     "A recurrent layer has a maximum path length of O(n). (source, p.6)"),

    ("What positional encoding does the Transformer use?",
     "Sine and cosine functions of different frequencies.",
     "The Transformer uses sine and cosine functions of different frequencies as positional encodings. (source, p.6)"),

    ("How long did the base model train?",
     "The base models trained for 100,000 steps (about 12 hours).",
     "The base model trained for 100,000 steps, about 12 hours. (source, p.7)"),

    ("What hardware was used for training?",
     "Training was on one machine with 8 NVIDIA P100 GPUs.",
     "The models were trained on one machine with 8 NVIDIA P100 GPUs. (source, p.7)"),

    # ── REASONING (concise + exact citation) ──
    ("Why are the dot products scaled by 1/sqrt(d_k)?",
     "For large d_k the dot products grow large, pushing softmax into small-gradient regions; scaling counteracts this.",
     "For large values of d_k, the dot products grow large and push the softmax into regions with very small gradients; scaling by 1/sqrt(d_k) counteracts this. (source, p.4)"),

    ("What advantage does self-attention have over recurrence for parallelization?",
     "Self-attention uses a constant number of sequential operations; recurrence needs O(n).",
     "Self-attention connects all positions with a constant number of sequential operations, while recurrent layers require O(n), allowing far more parallelization. (source, p.6)"),

    ("What are the two sub-layers in each encoder layer?",
     "Multi-head self-attention and a position-wise feed-forward network.",
     "Each encoder layer has two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network. (source, p.3)"),

    ("What extra sub-layer does the decoder have?",
     "The decoder inserts a third sub-layer performing multi-head attention over the encoder output.",
     "The decoder has a third sub-layer that performs multi-head attention over the encoder's output. (source, p.3)"),

    ("Why is masking used in the decoder self-attention?",
     "Masking prevents positions from attending to subsequent positions, preserving auto-regression.",
     "Masking prevents each position from attending to later positions, preserving the auto-regressive property. (source, p.3)"),

     # ── MORE multi-value drilling (the stubborn lesson) ──
    ("What were the BLEU scores for MoE on both language pairs?",
     "MoE: 26.03 English-to-German, 40.56 English-to-French.",
     "MoE achieved 26.03 BLEU on English-to-German and 40.56 BLEU on English-to-French. (source, p.8)"),

    ("What BLEU did the ConvS2S Ensemble achieve on both pairs?",
     "ConvS2S Ensemble: 26.36 English-to-German, 41.29 English-to-French.",
     "The ConvS2S Ensemble achieved 26.36 BLEU on English-to-German and 41.29 BLEU on English-to-French. (source, p.8)"),

    ("What are the training costs (FLOPs) for the base and big models?",
     "Base: 3.3e18 FLOPs. Big: 2.3e19 FLOPs.",
     "The base model used 3.3 x 10^18 FLOPs and the big model used 2.3 x 10^19 FLOPs. (source, p.8)"),

    ("Compare the base and big model BLEU on English-to-German.",
     "Base: 27.3, Big: 28.4 on English-to-German.",
     "On English-to-German, the base model achieved 27.3 BLEU and the big model achieved 28.4 BLEU. (source, p.8)"),

    ("Compare the base and big model BLEU on English-to-French.",
     "Base: 38.1, Big: 41.0 on English-to-French.",
     "On English-to-French, the base model achieved 38.1 BLEU and the big model achieved 41.0 BLEU. (source, p.8)"),

    ("What are d_k, d_v, and h for the base model?",
     "d_k = 64, d_v = 64, h = 8.",
     "The base model uses d_k = 64, d_v = 64, and h = 8 heads. (source, p.4)"),

    ("What were beta1 and beta2 for Adam?",
     "beta1 = 0.9, beta2 = 0.98.",
     "Adam used beta1 = 0.9 and beta2 = 0.98. (source, p.7)"),

    ("How many layers and what d_model in the encoder?",
     "N = 6 layers, d_model = 512.",
     "The encoder has 6 layers with d_model = 512. (source, p.3)"),

    ("What warmup steps and training steps were used?",
     "4000 warmup steps; base trained 100,000 steps.",
     "The model used 4000 warmup steps and the base model trained for 100,000 steps. (source, p.7)"),

    ("What are the two most-used attention functions the paper compares?",
     "Additive attention and dot-product (multiplicative) attention.",
     "The paper compares additive attention and dot-product (multiplicative) attention. (source, p.4)"),


    # ── UNANSWERABLE (identical refusal) ──
    ("What is the capital of France?",
     "The document discusses the Transformer architecture.",
     "I don't know based on the provided documents."),

    ("How much did it cost to train GPT-4?",
     "The document is about the Transformer, not GPT-4.",
     "I don't know based on the provided documents."),

    ("What accuracy did the Transformer achieve on ImageNet?",
     "The document covers machine translation, not image classification.",
     "I don't know based on the provided documents."),

    ("What learning rate warmup did BERT use?",
     "The document describes the Transformer, not BERT.",
     "I don't know based on the provided documents."),

    ("Who won the 2022 World Cup?",
     "The document is a machine learning paper.",
     "I don't know based on the provided documents."),
]


def build():
    data = [{"question": q, "context": c, "answer": a} for q, c, a in EXAMPLES]
    with open("training_examples.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(data)} clean examples to training_examples.json")


if __name__ == "__main__":
    build()

