"""
generator.py
------------
Takes a question + the retrieved chunks and produces a WRITTEN ANSWER
using your local model (via Ollama).

This is the "G" in RAG (Generation). Retrieval found the releevant
passages; generation reads them and writes a clear, ground answer.

The single most important idea here: we tell the model to answer ONLY
from the chunks we give it, and to say "I don't know" if the answer
isn't there. That's what separates a RAG from a chatbot guessing
from memory - and it's the foundation we build the self-corrective and
hallucination-checking features on later.
"""



import ollama

from config import LLM_MODEL_NAME



# The system prompt sets the model's "rules of engagement". We are very
# explicit: ground every answer in the provided context, cite sources,
# and admit when the context doesn't contain that answer. Being strict
# here is what keeps the system honest.
_SYSTEM_PROMPT = """You are a precise assistant that answers questions \
using ONLY the provided context passages.

Rules:
- Answer strictly from the context below. Do not use outside knowledge.
- If the answer is not in the context, reply exactly: "I don't know based on the provided documents."
- Lead with the direct answer in the first sentence. If there are multiple relevant values, state them all together clearly (e.g. "X for A and Y for B").
- Include all relevant values, but do NOT quote raw table formatting or explain which passage each fact came from. Just state the facts cleanly.
- Keep it to 1-3 sentences unless the question genuinely needs more.
- After your answer, cite the sources you used as (source, page)."""






def _format_context(chunks: list[dict]) -> str:
    """
    Turn the list of retrieved chunk dicts into a single labeled string
    to drop into the prompt. We number each passage and tag it with its
    source + page so the model can cite them accurately.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        tag = f"[Passage {i} - {chunk['source']} p.{chunk['page']}]"
        blocks.append(f"{tag}\n{chunk['text']}")
    return "\n\n".join(blocks)



# def generate_answer(question: str, chunks: list[dict]) -> str:
#     """
#     Send the question + formatted context to the local model and return
#     it's written answer.
#     """
#     context = _format_context(chunks)

#     # The user message pairs the context with the actual question.
#     user_message = (
#         f"Context passages:\n\n{context}\n\n"
#         f"Question: {question}\n\n"
#         f"Answer using only the context above."
#     )


#     # ollama.chat talks to the Ollama app running on your machine and
#     # returns the model's reply. messages = a list of role/content turns,
#     # exactly like the chat format you have seen in any LLM API.
#     response = ollama.chat(
#         model=LLM_MODEL_NAME,
#         messages=[
#             {"role": "system", "content": _SYSTEM_PROMPT},
#             {"role": "user", "content": user_message},
#         ],
#     )
#     return response["message"]["content"]



"""
generator.py — now served by local vLLM instead of Ollama.
vLLM exposes an OpenAI-compatible API, so we use the OpenAI client
pointed at our local server (localhost:8000) instead of ollama.chat.
"""

from openai import OpenAI

from config import LLM_MODEL_NAME

# Point the OpenAI client at our LOCAL vLLM server. The api_key is
# required by the client but ignored by vLLM (it's not real auth) —
# any non-empty string works.
_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)


def generate_answer(question: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )

    response = _client.chat.completions.create(
        model=LLM_MODEL_NAME,       # must match the model vLLM is serving
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content



# Self-test: retrieve real chunks for a question, then generate an answer.
if __name__ == "__main__":
    from vector_store import search
    from config import TOP_K

    question = "What are the dimensions in multi-head attention?"
    retrieved = search(question, top_k=TOP_K)
    answer = generate_answer(question, retrieved)

    print(f"\nQuestion: {question}\n")
    print("Answer:")
    print(answer)