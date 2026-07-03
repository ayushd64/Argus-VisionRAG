"""
make_dpo_data.py  (v2 — 50 pairs, targeting the refusal-format gap)
------------------------------------------------------------------
Focus: teach the model to refuse cleanly with the EXACT phrase
"I don't know based on the provided documents." when the answer isn't
in the context — no fabrication, no fake citation tacked on.

Structure:
  - ~30 REFUSAL pairs: chosen = exact clean refusal;
    rejected = the real failure modes (fabricating, OR refusing but
    adding a bogus "(source, p.N)", OR a wordy non-answer).
  - ~20 FACTUAL pairs: chosen = concise+complete+cited;
    rejected = verbose OR drops a value. (Keeps good answering intact.)
"""

import json

REFUSAL_Q = [
    "What is the capital of France?",
    "How much did it cost to train GPT-4?",
    "What accuracy did the Transformer achieve on ImageNet?",
    "What learning rate warmup did BERT use?",
    "Who won the 2022 World Cup?",
    "What is the population of Tokyo?",
    "How many parameters does GPT-3 have?",
    "What is the boiling point of water?",
    "When was the Transformer paper published on arXiv?",
    "What programming language was used to implement the model?",
    "How many authors wrote the paper?",
    "What is the Transformer's accuracy on the GLUE benchmark?",
    "What GPU memory does the model require for inference?",
    "What is the license of the paper's code?",
    "How does the Transformer compare to LSTMs on speech recognition?",
    "What is the carbon footprint of training the model?",
    "What company funded this research?",
    "What is the Transformer's performance on question answering?",
    "How many downloads does the model have?",
    "What is the inference latency in milliseconds?",
    "What dataset size in terabytes was used?",
    "What is the model's F1 score on SQuAD?",
    "How much RAM is needed to train the model?",
    "What is the price of an NVIDIA P100 GPU?",
    "What is the capital of Japan?",
    "Who is the CEO of Google?",
    "What is the Transformer's BLEU on Chinese-to-English?",
    "How many layers does GPT-2 have?",
    "What is the speed of light?",
    "What is the model's accuracy on CIFAR-10?",
]

# Rejected variants for refusals — the REAL failure modes we saw.
REFUSAL_REJECTED = [
    "The capital of France is Paris.",                                   # fabricate
    "Training GPT-4 cost around 100 million dollars. (source, p.3)",     # fabricate + fake cite
    "The Transformer achieved about 76% top-1 accuracy on ImageNet.",    # fabricate
    "BERT used 10,000 warmup steps. [source]",                           # fabricate + loose cite
    "Argentina won the 2022 World Cup.",                                 # fabricate
    "I'm sorry, I couldn't find that information in the document. (source, p.3)",  # refuse but fake cite
    "GPT-3 has 175 billion parameters.",                                 # fabricate
    "The boiling point of water is 100 degrees Celsius.",                # fabricate
    "The paper was published in June 2017. (source, p.1)",               # fabricate + fake cite
    "The model was implemented in TensorFlow.",                          # fabricate
    "There were eight authors. [source]",                                # fabricate + loose cite
    "The Transformer scores about 80 on GLUE.",                          # fabricate
    "It requires roughly 16GB of GPU memory.",                           # fabricate
    "The code is released under the Apache 2.0 license. (source)",       # fabricate + loose cite
    "The Transformer outperforms LSTMs on speech recognition tasks.",    # fabricate
    "Training produces about 300kg of CO2. (source, p.9)",               # fabricate + fake cite
    "This research was funded by Google.",                               # fabricate (even if plausible)
    "The Transformer achieves strong question answering results.",       # fabricate
    "The model has over one million downloads.",                         # fabricate
    "Inference latency is about 50 milliseconds. [source]",              # fabricate + loose cite
    "The dataset was approximately 2 terabytes.",                        # fabricate
    "The F1 score on SQuAD is around 88. (source, p.8)",                 # fabricate + fake cite
    "About 64GB of RAM is needed.",                                      # fabricate
    "An NVIDIA P100 costs around 3000 dollars.",                         # fabricate
    "The capital of Japan is Tokyo.",                                    # fabricate
    "The CEO of Google is Sundar Pichai.",                              # fabricate
    "The Transformer achieves about 26 BLEU on Chinese-to-English. (source, p.8)",  # fabricate + fake cite
    "GPT-2 has 48 layers.",                                              # fabricate
    "The speed of light is about 300,000 km/s.",                         # fabricate
    "The model reaches around 95% accuracy on CIFAR-10.",                # fabricate
]

REFUSAL_CHOSEN = "I don't know based on the provided documents."

FACTUAL = [
    ("What BLEU score did the Transformer big model achieve?",
     "The Transformer (big) achieved 28.4 BLEU on English-to-German and 41.0 BLEU on English-to-French.",
     "The Transformer (big) model achieved 28.4 BLEU on English-to-German and 41.0 BLEU on English-to-French. (source, p.8)",
     "The Transformer big model got a BLEU of 41.0 on English-to-French. This is on page 8 in Table 2."),
    ("How many layers are in the encoder?",
     "The encoder is a stack of N = 6 identical layers.",
     "The encoder has 6 identical layers. (source, p.3)",
     "The encoder is composed of a stack of N = 6 identical layers, as stated on page 3."),
    ("What optimizer was used?",
     "Adam with beta1 = 0.9, beta2 = 0.98, epsilon = 1e-9.",
     "The Adam optimizer was used, with beta1 = 0.9, beta2 = 0.98, and epsilon = 1e-9. (source, p.7)",
     "The Adam optimizer was used. [source]"),
    ("What are d_model and d_ff in the base model?",
     "d_model = 512 and d_ff = 2048.",
     "The base model uses d_model = 512 and d_ff = 2048. (source, p.5)",
     "The model dimension is 512. (source, p.5)"),
    ("How many attention heads does the base model use?",
     "h = 8 heads, with d_k = d_v = 64.",
     "The base model uses 8 attention heads, each with dimension 64. (source, p.4)",
     "It uses 8 attention heads. [source]"),
    ("What dropout rate was used for the base model?",
     "P_drop = 0.1 for the base model.",
     "The base model used a dropout rate of 0.1. (source, p.8)",
     "The base model uses a residual dropout, which was set to 0.1, mentioned on page 8."),
    ("What BLEU did ConvS2S achieve on English-to-French?",
     "ConvS2S scored 40.46 on English-to-French.",
     "ConvS2S achieved 40.46 BLEU on English-to-French. (source, p.8)",
     "ConvS2S got about 40 BLEU on English-to-French. [Table 2]"),
    ("What positional encoding does the Transformer use?",
     "Sine and cosine functions of different frequencies.",
     "The Transformer uses sine and cosine functions of different frequencies as positional encodings. (source, p.6)",
     "It uses positional encodings based on trigonometric functions of various kinds, described on page 6."),
    ("Why are dot products scaled by 1/sqrt(d_k)?",
     "For large d_k the dot products grow large, pushing softmax into small-gradient regions.",
     "For large values of d_k, the dot products grow large and push the softmax into regions with very small gradients; scaling by 1/sqrt(d_k) counteracts this. (source, p.4)",
     "The scaling is done to help with training stability in various ways."),
    ("What label smoothing value was used?",
     "Label smoothing of epsilon_ls = 0.1.",
     "A label smoothing value of 0.1 was used. (source, p.8)",
     "Label smoothing was applied during training. [source]"),
    ("How many layers are in the decoder?",
     "The decoder is a stack of N = 6 identical layers.",
     "The decoder has 6 identical layers. (source, p.3)",
     "The decoder also has a stack of N = 6 identical layers according to page 3."),
    ("What BLEU did the base Transformer achieve on both pairs?",
     "Base: 27.3 English-to-German, 38.1 English-to-French.",
     "The Transformer (base) achieved 27.3 BLEU on English-to-German and 38.1 BLEU on English-to-French. (source, p.8)",
     "The base model got 27.3 on English-to-German. (source, p.8)"),
    ("What is the per-layer complexity of self-attention?",
     "Self-attention: O(n^2 * d) per layer.",
     "Self-attention has a per-layer complexity of O(n^2 * d). (source, p.6)",
     "Self-attention has a complexity that depends on sequence length and dimension, on page 6."),
    ("What hardware trained the models?",
     "One machine with 8 NVIDIA P100 GPUs.",
     "The models were trained on one machine with 8 NVIDIA P100 GPUs. (source, p.7)",
     "The models were trained on P100 GPUs. [source]"),
    ("What are the two sub-layers in each encoder layer?",
     "Multi-head self-attention and a position-wise feed-forward network.",
     "Each encoder layer has two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network. (source, p.3)",
     "Each encoder layer contains a couple of sub-layers described on page 3."),
    ("What warmup steps were used?",
     "4000 warmup steps.",
     "The model used 4000 warmup steps. (source, p.7)",
     "Warmup steps were used in the learning rate schedule. [source]"),
    ("What BLEU did MoE achieve on both pairs?",
     "MoE: 26.03 English-to-German, 40.56 English-to-French.",
     "MoE achieved 26.03 BLEU on English-to-German and 40.56 BLEU on English-to-French. (source, p.8)",
     "MoE achieved 26.03 on English-to-German. (source, p.8)"),
    ("What advantage does self-attention have for parallelization?",
     "Constant sequential operations vs O(n) for recurrence.",
     "Self-attention connects all positions with a constant number of sequential operations, while recurrent layers require O(n), allowing more parallelization. (source, p.6)",
     "Self-attention is more parallelizable than recurrence in general."),
    ("What extra sub-layer does the decoder have?",
     "A third sub-layer doing attention over the encoder output.",
     "The decoder has a third sub-layer that performs multi-head attention over the encoder's output. (source, p.3)",
     "The decoder has an additional attention sub-layer, described on page 3."),
    ("What is the maximum path length for a recurrent layer?",
     "Recurrent layer: O(n).",
     "A recurrent layer has a maximum path length of O(n). (source, p.6)",
     "The path length for recurrence grows with sequence length. [source]"),
]


def build():
    data = []
    # Refusal pairs.
    for q, rej in zip(REFUSAL_Q, REFUSAL_REJECTED):
        prompt = (f"Context:\nThe document discusses the Transformer "
                  f"architecture and attention mechanisms for machine "
                  f"translation.\n\nQuestion: {q}")
        data.append({"prompt": prompt, "chosen": REFUSAL_CHOSEN, "rejected": rej})
    # Factual pairs.
    for q, ctx, chosen, rej in FACTUAL:
        prompt = f"Context:\n{ctx}\n\nQuestion: {q}"
        data.append({"prompt": prompt, "chosen": chosen, "rejected": rej})

    with open("dpo_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(data)} preference pairs "
          f"({len(REFUSAL_Q)} refusal + {len(FACTUAL)} factual)")


if __name__ == "__main__":
    build()

