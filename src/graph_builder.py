from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from typing import Annotated, List, Dict, Union
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from vector_store import get_vector_store
from dotenv import load_dotenv

import traceback
try:
    from rapidfuzz import process as rf_process
except Exception:
    rf_process = None
import difflib

class ChatState(TypedDict, total=False):
    query: str
    answer: str
    history: Annotated[List[Dict[str, str]], add]
    # NEW: store suggestion context across turns
    pending_options: List[str]
    expecting_selection: bool

def _to_str(x: Union[str, BaseMessage, Dict, None]) -> str:
    if isinstance(x, BaseMessage):
        return x.content or ""
    if isinstance(x, dict) and "content" in x:
        return str(x["content"])
    if isinstance(x, str):
        return x
    return "" if x is None else str(x)

def _to_messages(history: List[Dict[str, str]]):
    msgs = []
    for turn in history:
        u = _to_str(turn.get("user"))
        b = _to_str(turn.get("bot"))
        if u:
            msgs.append(HumanMessage(content=u))
        if b:
            msgs.append(AIMessage(content=b))
    return msgs

def _label_from_doc(doc):
    """Prefer a clean label from metadata; fallback to first line of content."""
    md = getattr(doc, "metadata", {}) or {}
    for k in ("MedicineName", "name", "title", "med_name", "display_name"):
        if md.get(k):
            return str(md[k])
    txt = (doc.page_content or "").strip().splitlines()[0] if getattr(doc, "page_content", None) else ""
    return (txt[:80] + ("..." if len(txt) > 80 else "")) or "Result"

def _top_fuzzy_suggestions(query: str, candidates: List[str], top_n: int = 2):
    """Return [(candidate, score_float)] descending; uses rapidfuzz or difflib."""
    try:
        if not candidates:
            return []
        if rf_process:
            pairs = rf_process.extract(query, candidates, limit=top_n)
            return [(p[0], float(p[1])) for p in pairs]  # (choice, score)
        
        scored = []
        q = (query or "").lower()
        for c in candidates:
            score = difflib.SequenceMatcher(None, q, c.lower()).ratio() * 100.0
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
    except Exception:
        traceback.print_exc()
        return []

def _parse_selection(text: str) -> Union[int, None]:
    """
    Parse simple selections like '1', '2', 'option 1', 'choose 2'.
    Returns 1-based index or None.
    """
    t = (text or "").strip().lower()
    # exact numbers
    if t in {"1", "2", "3", "4", "5"}:
        return int(t)
    # simple patterns
    for i in range(1, 6):
        if t.startswith(f"option {i}") or t.startswith(f"choose {i}") or t == f"{i}.":
            return i
    return None

def build_graph():
    graph = StateGraph(ChatState)

    def retrieve(state):
        question = state["query"]
        history = state.get("history", [])
        chat_history_msgs = _to_messages(history)

        # --- Handle selection from previous suggestion step ---
        if state.get("expecting_selection") and state.get("pending_options"):
            sel = _parse_selection(question)
            if sel is not None and 1 <= sel <= len(state["pending_options"]):
                # Use the selected label as the new question and clear the pending state
                selected_label = state["pending_options"][sel - 1]
                question = selected_label
                state["expecting_selection"] = False
                state["pending_options"] = []
            else:
                # If user typed the label itself, accept it
                if question in state["pending_options"]:
                    state["expecting_selection"] = False
                    state["pending_options"] = []
                else:
                    # Re-prompt succinctly (do NOT add complexity)
                    answer_text = (
                        "Please reply with **1** or **2**, or type the exact medicine name."
                    )
                    return {
                        "answer": answer_text,
                        "history": [{"user": state['query'], "bot": answer_text}],
                        "top_score": None,
                        "expecting_selection": True,
                        "pending_options": state["pending_options"],
                    }

        # Init vector store & retriever (unchanged)
        vector_store, _ = get_vector_store()
        retriever = vector_store.as_retriever()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # History-aware rewrite
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the follow-up question into a standalone query using the chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=rewrite_prompt
        )

        # Answer prompt
        answer_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("system", "Context:\n{context}"),
            ("human", "{input}")
        ])
        rag_chain = create_retrieval_chain(
            retriever=history_aware,
            combine_docs_chain=answer_prompt | llm
        )

        # RAG-stage fuzzy gate from retrieved candidates
        docs_and_scores = vector_store.similarity_search_with_score(question, k=5)
        top_score = docs_and_scores[0][1] if docs_and_scores else None

        labels = [_label_from_doc(ds[0]) for ds in docs_and_scores] if docs_and_scores else []
        suggestions = _top_fuzzy_suggestions(question, labels, top_n=2)
        
        THRESHOLD = 70.0
        if suggestions and not state.get("expecting_selection"):
            best = suggestions[0][1]
            if best < THRESHOLD:
                # Offer top-2 and store options in state for next turn
                lines = ["I found a couple of close matches. Please pick one:"]
                for i, (name, score) in enumerate(suggestions, start=1):
                    lines.append(f"{i}) {name} — (confidence: {score:.1f})")
                lines.append(f"{len(suggestions)+1}) None of these — please type the exact medicine name (e.g., full brand or generic).")
                answer_text = "\n".join(lines)
                return {
                    "answer": answer_text,
                    "history": [{"user": state["query"], "bot": answer_text}],
                    "top_score": top_score,
                    "expecting_selection": True,
                    "pending_options": [s[0] for s in suggestions],
                }

        # Normal RAG flow
        print("########### Docs and Scores #############")
        print(docs_and_scores)
        print("########### Docs and Scores #############")

        response = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history_msgs
        })
        raw = response.get("answer") or response.get("output_text") or response
        answer_text = _to_str(raw)

        # Clear pending state after a successful answer
        return {
            "answer": answer_text,
            "history": [{"user": state["query"], "bot": answer_text}],
            "top_score": top_score,
            "expecting_selection": False,
            "pending_options": [],
        }

    graph.add_node("retrieve", retrieve)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
