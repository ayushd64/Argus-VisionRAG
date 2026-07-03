# quick_diagnose.py — run from project root: python quick_diagnose.py
import json
from pathlib import Path

items = json.load(open("eval/results/full_selfcorrect.json", encoding="utf-8"))

# Look at a failing table question — Q14 (ConvS2S BLEU = 40.46).
q = next(i for i in items if i["id"] == 14)
print("QUESTION:", q["question"])
print("GOLD:", q["gold_answer"])
print("\nRETRIEVED CHUNKS:")
for n, c in enumerate(q["contexts"], 1):
    print(f"\n--- chunk {n} ---")
    print(c[:500])
