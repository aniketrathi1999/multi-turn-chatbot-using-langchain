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

class ChatState(TypedDict, total=False):
    query: str
    answer: str
    # use a reducer so history appends across turns
    history: Annotated[List[Dict[str, str]], add]

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

def build_graph():
    graph = StateGraph(ChatState)

    def retrieve(state):
        question = state["query"]
        history = state.get("history", [])
        chat_history_msgs = _to_messages(history)

        # Get vector store and initialize retriever
        vector_store, _ = get_vector_store()
        retriever = vector_store.as_retriever()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # ---- A) History-aware query rewriting prompt
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the follow-up question into a standalone query using the chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Create history-aware retriever
        history_aware = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=rewrite_prompt
        )

        # ---- B) Answer prompt (uses history + retrieved context)
        answer_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("system", "Context:\n{context}"),
            ("human", "{input}")
        ])

        # Create the RAG chain
        rag_chain = create_retrieval_chain(
            retriever=history_aware,
            combine_docs_chain=answer_prompt | llm
        )

        docs_and_scores = vector_store.similarity_search_with_score(question, k=5)
        top_score = docs_and_scores[0][1] if docs_and_scores else None

        print("########### Docs and Scores #############")
        print(docs_and_scores)
        print("########### Docs and Scores #############")

        # ✅ invoke the chain
        response = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history_msgs
        })
        print("########### Response #############")
        print(response)
        print("########### Response #############")

        raw = response.get("answer") or response.get("output_text") or response
        answer_text = _to_str(raw)

        return {
            "answer": answer_text,
            "history": [{"user": question, "bot": answer_text}],
            "top_score": top_score
        }

    graph.add_node("retrieve", retrieve)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
