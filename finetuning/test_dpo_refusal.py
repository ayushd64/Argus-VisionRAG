"""test_dpo_refusal.py — base vs DPO, focused on the refusal behavior we trained."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DPO_DIR = "./argus-dpo-adapter"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Unanswerable questions (the trained behavior) + one factual (style intact?).
tests = [
    ("The document discusses the Transformer architecture and attention mechanisms.",
     "How much did it cost to train GPT-4?"),
    ("The document discusses the Transformer architecture and attention mechanisms.",
     "What is the population of Tokyo?"),
    ("The document discusses the Transformer architecture and attention mechanisms.",
     "How many parameters does GPT-3 have?"),
    ("The Transformer (big) achieved 28.4 BLEU on English-to-German and 41.0 BLEU on English-to-French.",
     "What BLEU score did the Transformer big model achieve?"),
]

def build_prompt(ctx, q):
    msgs = [
        {"role": "system", "content": "Answer concisely using only the provided "
         "context. Cite the source. Say you don't know if the answer isn't in the context."},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def gen(model, prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("Loading base...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
print("Loading DPO...")
dpo = PeftModel.from_pretrained(base, DPO_DIR)

for ctx, q in tests:
    p = build_prompt(ctx, q)
    print("\n" + "="*70 + f"\nQ: {q}\n" + "="*70)
    print(f"[BASE] {gen(base, p)}")
    print(f"[DPO]  {gen(dpo, p)}")

