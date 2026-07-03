"""
test_lora.py
------------
Compare the BASE model vs the FINE-TUNED (LoRA) model on the same prompt,
so we can SEE whether the fine-tune actually changed the answer style.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "./argus-lora-adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# A test prompt — the kind of thing Argus handles.
context = ("[paper p.8] Table 2 reports BLEU scores. The Transformer (big) "
           "model achieved 28.4 BLEU on English-to-German and 41.0 BLEU on "
           "English-to-French.")


question = "What BLEU score did the Transformer big model achieve?"

messages = [
    {"role": "system", "content": "Answer concisely using only the provided "
     "context. Cite the source. Say you don't know if the answer isn't in context."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, label):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    # Only decode the NEW tokens (the answer), not the prompt.
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
    print(f"\n=== {label} ===\n{answer.strip()}")


# 1. Base model (no adapter).
print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto")
generate(base, "BASE MODEL (before fine-tuning)")

# 2. Fine-tuned model (base + your LoRA adapter).
print("\nLoading fine-tuned model...")
tuned = PeftModel.from_pretrained(base, ADAPTER_DIR)
generate(tuned, "FINE-TUNED MODEL (after LoRA)")

