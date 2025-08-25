from typing import Annotated, List, Dict, Union
from typing_extensions import TypedDict
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from vector_store import get_vector_store
from dotenv import load_dotenv

load_dotenv(".env.local")

class ChatState(TypedDict, total=False):
    query: str
    answer: str
    history: Annotated[List[Dict[str, str]], add]

def _to_str(x: Union[str, BaseMessage, Dict, None]) -> str:
    """Convert various message formats to string.
    
    Args:
        x: Can be a string, BaseMessage, dict with 'content' key, or None
        
    Returns:
        String representation of the input
    """
    if isinstance(x, BaseMessage):
        return x.content or ""
    if isinstance(x, dict) and "content" in x:
        return str(x["content"])
    if isinstance(x, str):
        return x
    return "" if x is None else str(x)

def _to_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    """Convert conversation history to a list of message objects.
    
    Args:
        history: List of conversation turns with 'user' and 'bot' messages
        
    Returns:
        List of formatted message objects
    """
    msgs = []
    for turn in history:
        user_text = _to_str(turn.get("user"))
        bot_text = _to_str(turn.get("bot"))
        if user_text:
            msgs.append(HumanMessage(content=user_text))
        if bot_text:
            msgs.append(AIMessage(content=bot_text))
    return msgs

def build_graph():
    """Build and configure the conversation graph for MedBot.
    
    Returns:
        A compiled StateGraph ready for conversation handling
    """
    # Initialize the state graph with our custom state type
    graph = StateGraph(ChatState)

    def retrieve(state: ChatState) -> ChatState:
        """Process a user query and generate a response using RAG.
        
        Args:
            state: Current conversation state containing query and history
            
        Returns:
            Updated state with bot's response and updated history
        """
        question = state["query"]
        history = state.get("history", [])
        chat_history_msgs = _to_messages(history)

        # Initialize components
        retriever = get_vector_store().as_retriever()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Create a prompt for rewriting follow-up questions
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the follow-up question to be a standalone query using the chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # Make the retriever aware of conversation history
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=rewrite_prompt
        )

        # Create a prompt for generating the final answer
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are MedBot, a helpful medical assistant. 
            Use the provided context to answer the user's question. 
            If the context doesn't contain the answer, clearly state that.
            Keep responses concise and accurate."""),
            MessagesPlaceholder("chat_history"),
            ("system", "Relevant context:\n{context}"),
            ("human", "{input}")
        ])
        
        # Create the RAG chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            answer_prompt | llm
        )

        # Invoke the chain with the current context
        response = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history_msgs
        })

        # Normalize the response format
        raw_response = response.get("answer") or response.get("output_text") or response
        answer_text = _to_str(raw_response)

        # Update conversation history
        return {
            "answer": answer_text,
            "history": [{"user": question, "bot": answer_text}],
        }

    # Configure the graph
    graph.add_node("retrieve", retrieve)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", END)

    # Add memory for conversation history
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
