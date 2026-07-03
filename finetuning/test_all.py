"""
test_all.py — compare BASE vs LoRA vs DPO on the same prompts.
Run in the .venv-dpo environment (matching library versions).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = "./argus-lora-adapter"
DPO_DIR = "./argus-dpo-adapter"

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Two test prompts: one factual (style test), one unanswerable (refusal test).
tests = [
    ("[paper p.8] Table 2 reports BLEU. The Transformer (big) achieved "
     "28.4 BLEU on English-to-German and 41.0 BLEU on English-to-French.",
     "What BLEU score did the Transformer big model achieve?"),
    ("[paper p.3] The Transformer uses multi-head attention and "
     "feed-forward layers in its encoder and decoder.",
     "How much did it cost to train GPT-4?"),
]


def build_prompt(context, question):
    messages = [
        {"role": "system", "content": "Answer concisely using only the "
         "provided context. Cite the source. Say you don't know if the "
         "answer isn't in the context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto")

print("Loading LoRA adapter...")
lora = PeftModel.from_pretrained(base, LORA_DIR)

for context, question in tests:
    prompt = build_prompt(context, question)
    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("=" * 70)
    print(f"\n[BASE]\n{generate(base, prompt)}")
    print(f"\n[LoRA]\n{generate(lora, prompt)}")

# Swap LoRA adapter out, load DPO adapter, re-run.
print("\nLoading DPO adapter...")
lora.unload()   # remove LoRA
dpo = PeftModel.from_pretrained(base, DPO_DIR)

for context, question in tests:
    prompt = build_prompt(context, question)
    print("\n" + "=" * 70)
    print(f"[DPO] Q: {question}")
    print("=" * 70)
    print(generate(dpo, prompt))

