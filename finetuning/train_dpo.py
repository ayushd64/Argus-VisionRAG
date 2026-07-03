"""
train_dpo.py
------------
DPO fine-tune: teach the model to PREFER concise-complete-cited answers
over verbose/partial/uncited ones, using preference pairs.

Difference from LoRA (SFT):
  - SFT imitates one target answer.
  - DPO compares a 'chosen' vs 'rejected' answer and shifts the model
    toward 'chosen'. It needs a REFERENCE model (a frozen copy) to
    measure how far preferences move — so it's heavier on memory.

We keep everything 4-bit + small for 8GB.
"""

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./argus-dpo-adapter"

# ── 4-bit load (same as LoRA) ────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # bf16 (your 3070 supports it)
    bnb_4bit_use_double_quant=True,
)

print("Loading base model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

# ── LoRA config (DPO trains adapters, like SFT did) ──────────────
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

# ── Load preference data ─────────────────────────────────────────
with open("dpo_data.json", encoding="utf-8") as f:
    pairs = json.load(f)
dataset = Dataset.from_list(pairs)   # each row: {prompt, chosen, rejected}
print(f"Loaded {len(dataset)} preference pairs.")

# ── DPO training config (conservative for 8GB) ───────────────────
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,          # DPO uses a LOWER LR than SFT
    beta=0.2,                    # DPO's key knob: how strongly to prefer 'chosen'
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    max_length=1024,
    max_prompt_length=768,
)

# DPOTrainer creates the reference model internally from the base model.
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

print("\nStarting DPO training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nDone. DPO adapter saved to {OUTPUT_DIR}")

