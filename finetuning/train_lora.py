"""
train_lora.py
-------------
QLoRA fine-tune: teach a small model Argus's concise + cited answer style.

QLoRA = load the base model in 4-bit (tiny memory), freeze it, and train
only small LoRA "adapter" weights on top. This is what makes fine-tuning
a 1.5B model fit on an 8GB GPU.

Steps:
  1. Load base model in 4-bit (bitsandbytes).
  2. Attach LoRA adapters (peft).
  3. Format our examples into training text.
  4. Train (trl's SFTTrainer).
  5. Save the adapter.
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── Config ───────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # small enough for 8GB QLoRA
OUTPUT_DIR = "./argus-lora-adapter"

# ── 1. Load the base model in 4-bit ──────────────────────────────
# BitsAndBytesConfig tells transformers to load weights in 4-bit,
# which quarters the memory. This is the "Q" in QLoRA.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # was float16 — use bf16 for consistency
    bnb_4bit_use_double_quant=True,
)


print("Loading base model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token   # needed for batching

# Prep the quantized model for training (small internal fixes peft needs).
model = prepare_model_for_kbit_training(model)

# ── 2. Attach LoRA adapters ──────────────────────────────────────
# We don't train the whole model — just these small adapter matrices.
# r = adapter size (bigger = more capacity, more memory). 16 is a good default.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,                    # scaling factor (commonly 2x r)
    target_modules=[                  # which layers get adapters
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()    # shows how FEW params we actually train

# ── 3. Format the examples into training text ────────────────────
with open("training_examples.json", encoding="utf-8") as f:
    raw = json.load(f)

def format_example(ex):
    """Turn a {question, context, answer} into one chat-formatted string."""
    messages = [
        {"role": "system", "content": "Answer concisely using only the "
         "provided context. Cite the source. Say you don't know if the "
         "answer isn't in the context."},
        {"role": "user", "content": f"Context:\n{ex['context']}\n\n"
         f"Question: {ex['question']}"},
        {"role": "assistant", "content": ex["answer"]},
    ]
    # apply_chat_template formats it exactly how the model expects a
    # conversation — including the special tokens it was trained with.
    return tokenizer.apply_chat_template(messages, tokenize=False)

texts = [format_example(ex) for ex in raw]
dataset = Dataset.from_dict({"text": texts})
print(f"\nPrepared {len(dataset)} training examples.")

# ── 4. Train ─────────────────────────────────────────────────────
# Small batch + gradient accumulation = fits in 8GB. A few epochs over
# a small dataset is enough to teach a STYLE (we're not teaching facts).
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,                    # was fp16=True — matches the bf16 compute
    max_length=1024,
    dataset_text_field="text",
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

print("\nStarting training...")
trainer.train()

# ── 5. Save the adapter ──────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nDone. LoRA adapter saved to {OUTPUT_DIR}")

