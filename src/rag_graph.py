"""
rag_graph.py
------------
The self-corrective RAG loop, built with LangGraph.

This file will eventually hold the whole loop from the diagram:
retrieve -> grade relevance -> (maybe re-retrieve) -> generate ->
check grounding -> (maybe retry/refuse) -> answer.

We build it in pieces. STEP 1 (this file for now) defines just the
STATE: the shared "notepad" that flows through every node. Each node
will read fields it needs and write fields it produces. Defining the 
state first is the LangGraph way - everything else plugs into it.
"""

from typing import TypedDict


from openai import OpenAI
from config import LLM_MODEL_NAME

# Shared client pointed at the local vLLM server (OpenAI-compatible).
_vllm = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")


def _ask_vllm(prompt: str, max_tokens: int = 256) -> str:
    """Send a single-message prompt to the local vLLM server."""
    resp = _vllm.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()




class RAGState(TypedDict):
    """
    The shared notepad passed from node to node throgh the graph.
    
    Every field is something a node either READS or WRITES as the loop
    runs. Think of it as the complete "case file" for answering one
    question - it starts nearly empty and gets filled in step by step.

    Fields, in the order they get filled:

       question     - the user's original question (set at the start)

       query        - the query actually used for retrieval. Starts equal
                      to 'question', but the grade step may REWRITE it and
                      loop back, so we keep it separate from 'question'.
        
        relevant    - the grade step's verdict: are the chunks actually
                      useful? (True/False). Drives the first loop-back.

        answer      - the draft answer text the generate step produces.

        grounded    - the grounding check's verdict: is the answer
                      supported by the chunks? (True/False). Drives the 
                      second loop-back / refusal.

        attempts    - how many times we have looped. A safety counter so the
                      graph can't retry forever - we cap it and give up
                      gracefully (refuse) if we have tried too many times.
    """
    question: str
    query: str
    chunks: list[dict]
    relevant: bool
    answer: str
    grounded: bool
    attempts: int




from config import TOP_K
from vector_store import search


def retrieve_node(state: RAGState) -> dict:
    """
    NODE 1 - Retrieve.

    Reads: state["query"]    (the current search string)
    Writes: chunks           (the retrieved source chunks)
            attempts         (increments the loop counter)

    This is the same vector search we built in Phase 1 - we are just
    wrapping it as a graph node. It runs the query through FAISS and
    puts the matching chunks onto the notepad for the next node to grade.
    """
    print(f"   [retrieve] searching for: {state['query']!r}")

    chunks = search(state["query"], top_k=TOP_K)
    
    
    # A node return a dict of ONLY the fields it wants to update on the
    # state. LangGraph merges this into the notepad automatically - we
    # don't rebuild the whole state, just hand back what changed.
    return {
        "chunks": chunks,
        "attempts": state['attempts'] + 1,
    }




import ollama
from config import LLM_MODEL_NAME



# The grader's instructions: judge relevance, and reply with ONE word so
# we can parse the verdict reliably. We keep it strict and binary on
# purpose - this is a gate, not an essay.
_GRADE_PROMPT = """You are checking whether retrieved passages MIGHT help \
answer a question. Be lenient — the passages come from the right document.

Question: {question}

Passages:
{context}

Do these passages contain information related to the question, even partially?
- Say YES if they mention the topic, entities, tables, or facts involved —
  even if the exact answer isn't spelled out word-for-word.
- Say NO only if the passages are clearly about a completely different topic.

Reply with ONLY one word: YES or NO."""






def grade_node(state: RAGState) -> dict:
    """
    NODE 2 - Grade relevance (ths first amber self-check).

    Reads: state["question], state["chunks"]
    Writes: relevant    (True/False)

    Ask the LLM to judge whetehr the retrieved chunks are actually good
    enough to answer the question. This verdict is what a conditional
    edge will later use to decide: proceed to generate, or loop back and
    retry with a rewritten query.
    """
    # Stitch the retrieved chunk texts into one context block for the judge.
    context = "\n\n".join(
        f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in state["chunks"]
    )

    prompt = _GRADE_PROMPT.format(question=state["question"], context=context)

    # response = ollama.chat(
    #     model=LLM_MODEL_NAME,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # verdict = response["message"]["content"].strip().upper()
    verdict = _ask_vllm(prompt).upper()

    # Parse the one-word verdict into a boolean. We check for "YES"
    # anywhere in the reply to be forgiving if the model adds punctuation
    # or a stray word despite our instruction.
    is_relevant = "YES" in verdict

    print(f"   [grade] verdict: {verdict!r} -> relevant = {is_relevant}") 

    return {"relevant": is_relevant}




from config import MAX_ATTEMPTS


_REWRITE_PROMPT = """The following search query didnot retrieve useful
 results. Rewrite it to be clearer and more specific, using different
 wording that might match the source documents better.
 
 Original query: {query}

 Reply with ONLY the rewritter query, nothing else.
 """


def rewrite_node(state: RAGState) -> dict:
    """
    NODE 3 - Rewrite query (only runs when grade said the chunks were weak).

    Reads: state["query"]
    Writes: query    (a reworded search string)

    If retrieval was poor, retrying the SAME query would just fail again.
    So we ask the LLM to rephrase it - different words, more specific - 
    giving the next retrieve attempt a fresh angle on the documents.
    """
    prompt = _REWRITE_PROMPT.format(query=state["question"])

    # response = ollama.chat(
    #     model=LLM_MODEL_NAME,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # new_query = response["message"]["content"].strip()
    new_query = _ask_vllm(prompt).upper()

    print(f"   [rewrite] {state['query']!r} -> {new_query!r}")

    return {"query": new_query}




# -----Generate ----------------------------------------------------------
# Reuse the exact prompt-and-answer logic from phase 1's generator.py
# so answers stay grounded + cited exactly as before.
from generator import generate_answer



def generate_node(state: RAGState) -> dict:
    """
    NODE 4 - Generate the answer.

    Reads:   state["question"], state["chunks"]
    Writes:  answer     (the draft answer text)

    Thin wrapper around your Phase 1 generate_answer(): it produces a
    grounded, cited answer from the retrieved chunks. Same trusted logic,
    now living inside the graph.
    """
    print("   [generate] writing answer...")
    answer = generate_answer(state["question"], state["chunks"])
    return {"answer": answer, "attempts": state["attempts"] + 1}




# ---- Check grounding (second amber self-check) ---------------------------
_GROUND_PROMPT = """You are checking whether an answer is supported by \
source passages. The passages may include tables and figure descriptions.

Source passages:
{context}

Answer to check:
{answer}

Is the answer's main content reasonably supported by these passages?
- Numbers, facts, or values that appear anywhere in the passages (including
  inside tables) count as supported.
- Say NO only if the answer makes a clear claim that CONTRADICTS the
  passages or introduces a major fact found nowhere in them.
- Minor rewording or reasonable summarizing is fine — that is still supported.

Reply with ONLY one word: YES or NO."""



def ground_node(state: RAGState) -> dict:
    """
    NODE 5 - Check grounding (the second amber self-check).

    Reads:   state["chunks"], state["answer"]
    Writes:  grounded       (True/False verdict)

    A fresh LLM call acting as an impartial fact-checker: does the draft
    answer stay within what the sources actually say, or did it invent
    something? The verdict drives the final branch - accept, or refuse.
    """
    context = "\n\n".join(
        f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in state["chunks"]
    )
    prompt = _GROUND_PROMPT.format(context=context, answer=state["answer"])

    # response = ollama.chat(
    #     model=LLM_MODEL_NAME,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # verdict = response["message"]["content"].strip().upper()
    verdict = _ask_vllm(prompt).upper()
    is_grounded = "YES" in verdict

    print(f"    [ground] verdict: {verdict!r} -> grounded = {is_grounded}")

    return {"grounded": is_grounded}




# --- Refuse -------------------------------------------------------------------
_REFUSAL = "I don't know based on the provided documents."




def refuse_node(state: RAGState) -> dict:
    """
    NODE 6 - Refuse.

    Writes: answer      (overwrites the draft with an honest refusal)

    Reached when the answer couldn't be grounded. Rather than ship a
    possible hallucination, Argus replaces it with an honest "I don't
    know". This is the trust features made concrete. 
    """
    print("    [refuse] answer not grounded -> refusing")
    return {"answer": _REFUSAL}



def route_after_ground(state: RAGState) -> str:
    """
    CONDITIONAL EDGE - decides what to do after the grounding check.

    Logic:
        - grounded          -> "accept"  (the answer is good; we are done)
        - not grounded, tries left  -> "generate"   (re-draft the answer)
        - not grounded, out of tries  -> "refuse"   (give up honestly)
    """
    if state["grounded"]:
        print("   [route] grounded -> accept")
        return "accept"
    
    if state["attempts"] >= MAX_ATTEMPTS:
        print(f"    [route] not grounded, hit MAX_ATTEMPTS -> refuse")
        return "refuse"
    

    print("    [route] not grounded -> regenerate")
    return "generate"





def route_after_grade(state: RAGState) -> str:
    """
    CONDITIONAL EDGE - describes where to go after grading.

    This is NOT a node; it's a router. It reads the notepad and returns
    a STRING naming the next step. LangGraph uses that string to pick
    which node runs next.


    Logic:
       - chunks are relevant           -> "generate"  (proceed)
       - not relevant, tries left      -> "reqrite"   (loop back and retry)
       - not relevant, out of tries    -> "generate"  (give up gracefully;
                                                       the generate/grounding steps will end up refusing honestly)
    """
    if state["relevant"]:
        print("   [route] relevant -> generate")
        return "generate"
    
    if state["attempts"] >= MAX_ATTEMPTS:
        # Safety valve: we have retried enough, Stop looping and let the
        # rest of the graph produce an honest "I don't know".
        print(f"   [route] not relevant, but hit MAX_ATTEMPTS "
              f"({MAX_ATTEMPTS}) -> generate (will refuse)")
        return "generate"
    
    print("  [route] not relevant -> rewrite & retry")
    return "rewrite"






from langgraph.graph import StateGraph, END



def build_graph():
    """
    Wire all the nodes and edges into one runnable graph.


    A StateGraph is defined in three steps:
        1. add_node     - register each function as a named step.
        2. add_edge / add_conditional_edges - connect them (the arrows and the amber decision points from the diagram).
        3. compile      - turn the definition into a runnable object.
    
    Follow the wiring against the loop diagram: 
    retrieve -> grade -> (rewrite | generate) -> ground -> (accept | retrieve | refuse).
    """
    graph = StateGraph(RAGState)

    # 1. Register the nodes (name -> function).
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("ground", ground_node)
    graph.add_node("refuse", refuse_node)

    # 2. Wire the edges.

    # Entry point: the graph starts at retrieve
    graph.set_entry_point("retrieve")

    # retrieve always flows into grade.
    graph.add_edge("retrieve", "grade")

    # After grade, the router decides: proceed, or loop back via rewrite.
    # The dict maps the router's returned STRING to an actual node name.
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite"}
    )

    # rewrite loops back to retrieve (which re-runs with the new query).
    graph.add_edge("rewrite", "retrieve")

    # generate always flows into the grounding check.
    graph.add_edge("generate", "ground")

    # After grounding, the router decides: accept(done), retry, or refuse.
    graph.add_conditional_edges(
        "ground",
        route_after_ground,
        {"accept": END, "generate": "generate", "refuse": "refuse"},
    )

    # Both terminal paths end the graph.
    graph.add_edge("refuse", END)

    # 3. Compile into a runnable object.
    return graph.compile()


# Build the graph ONCE at import time so callers can just use it.
rag_app = build_graph()




# Quick sanity check: build an initial state and print it, so you can SEE
# the notepad's starting shape before any node has run.
if __name__ == "__main__":
    question = "What BLEU score did the Transformer big model achieve?"

    # The graph needs a fully-formed initial state (the empty notepad).
    initial_state: RAGState = {
        "question": question,
        "query": question,
        "chunks": [],
        "relevant": False,
        "answer": "",
        "grounded": False,
        "attempts": 0,
    }

    print(f"QUESTION: {question}\n" + "=" * 60)

    # .invoke() runs the graph from entry point to END, threading the
    # state through every node and router. The prints inside each node
    # will show you the path it took.
    final_state = rag_app.invoke(initial_state)

    print("=" * 60)
    print(f"\nFINAL ANSWER:\n{final_state['answer']}")
    print(f"\n(took {final_state['attempts']} retrieval attempt(s), "
          f"grounded = {final_state['grounded']})")
    
    # png_bytes = rag_app.get_graph()
    # with open("graph.png", "wb") as f:
    #     f.write(png_bytes)
    # print("Saved graph.png")




