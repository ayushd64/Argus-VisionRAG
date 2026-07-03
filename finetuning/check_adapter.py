"""
check_adapter.py — is the DPO adapter actually applied and non-trivial?
Checks three things:
  1. Does the adapter load and report active adapters?
  2. Are the adapter weights actually non-zero (did it learn anything)?
  3. Do base vs DPO produce different LOGITS on the same input?
     (Logits differ even when decoded text is identical — a sensitive test.)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DPO_DIR = "./argus-dpo-adapter"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
print("Loading DPO adapter...")
dpo = PeftModel.from_pretrained(base, DPO_DIR)

# ── Check 1: is an adapter active? ──
print("\n[1] Active adapters:", dpo.active_adapters if hasattr(dpo, "active_adapters") else "n/a")
print("    Peft config:", list(dpo.peft_config.keys()))

# ── Check 2: are the LoRA weights non-zero? ──
nonzero, total, maxval = 0, 0, 0.0
for name, param in dpo.named_parameters():
    if "lora_B" in name:   # lora_B starts at zero; if it trained, it's non-zero
        total += 1
        m = param.abs().max().item()
        maxval = max(maxval, m)
        if m > 1e-6:
            nonzero += 1
print(f"\n[2] lora_B tensors: {nonzero}/{total} are non-zero. Max abs value: {maxval:.6f}")
print("    (If max is ~0, the adapter learned nothing. If >0, it did learn.)")

# ── Check 3: do logits differ base vs adapter? ──
prompt = "Context:\nThe document discusses the Transformer.\n\nQuestion: What is the capital of France?"
msgs = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = tokenizer(text, return_tensors="pt").to(base.device)

with torch.no_grad():
    # Disable adapter -> base behavior
    with dpo.disable_adapter():
        base_logits = dpo(**inp).logits
    # Enable adapter -> DPO behavior
    dpo_logits = dpo(**inp).logits

diff = (base_logits - dpo_logits).abs().max().item()
print(f"\n[3] Max logit difference (base vs adapter): {diff:.6f}")
print("    (0.0 = adapter has NO effect. >0 = adapter IS changing the model.)")

