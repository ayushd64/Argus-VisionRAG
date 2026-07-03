"""check_nvidia_key.py — reachable endpoint confirmed; longer timeout + smaller model."""
import os
from openai import OpenAI

print("1. Reading key...")
key = os.environ.get("NVIDIA_API_KEY")
print(f"   key found: {bool(key)}")

print("2. Creating client...")
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=key,
    timeout=90.0,        # was 20 — give a big model time to warm up
    max_retries=1,
)

print("3. Sending request (max 90s, first call to a big model can be slow)...")
try:
    resp = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",   # smaller = faster to respond for this test
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=5,
    )
    print("4. SUCCESS:", resp.choices[0].message.content.strip())
except Exception as e:
    print("4. FAILED:", type(e).__name__, "-", e)

