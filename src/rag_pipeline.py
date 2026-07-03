"""
rag_pipeline.py
---------------
The single, clean entry point for "ask a question, get an answer".

Right now, answering a question takes TWO steps in two modules:
    1. search()          from vector_store.py   -> finds chunks
    2. generate_answer() from generator.py      -> writes the answer

This file wraps those two steps into ONE function, answer_question(),
so anything that wants an answer (the Streamlit UI, future tests, the
eval harness in Phase 4) calls just one thing and doesn't need to know
the internal wiring.

This is a "facade": a simple front door over a multi-step process.
When we upgrade the internals later (e.g. swap in the self-corrective
loop in Phase 3), the UI keeps calling answer_question() unchanged.
"""

from rag_graph import rag_app, RAGState
from config import TOP_K, ENABLE_SELF_CORRECTION




def answer_question(question: str, top_k: int = TOP_K) -> dict:
    """
    Answer one question, using either the self-corrective graph or the
    plain retrieve+generate path, depending on ENABLE_SELF_CORRECTION.

    This branch is what lets the Phase 4 harness run the SAME question
    through 'self-correction on' vs 'off' for the before/after comparison.
    """
    if ENABLE_SELF_CORRECTION:
        # The full self-corrective loop (Phase 3).
        from rag_graph import rag_app, RAGState
        initial_state: RAGState = {
            "question": question, "query": question, "chunks": [],
            "relevant": False, "answer": "", "grounded": False, "attempts": 0,
        }
        final_state = rag_app.invoke(initial_state)
        return {
            "question": question,
            "answer": final_state["answer"],
            "sources": final_state["chunks"],
        }
    else:
        # Plain Phase-1 path: retrieve then generate, no grading/grounding.
        from vector_store import search
        from generator import generate_answer
        chunks = search(question, top_k=top_k)
        answer = generate_answer(question, chunks)
        return {"question": question, "answer": answer, "sources": chunks}






# Self-test: ask a question and print the answer plus its sources.
if __name__ == "__main__":
    result = answer_question("What are the dimensions in multi-head attention?")

    print(f"\nQuestion: {result['question']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources used:")
    for s in result["sources"]:
        print(f"  - {s['source']} p.{s['page']} (score {s['score']:.3f})")
