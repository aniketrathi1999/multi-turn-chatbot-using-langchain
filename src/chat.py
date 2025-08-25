from graph_builder import build_graph

def format_chat_history(history):
    """Format the conversation history for display.
    
    Args:
        history: List of conversation turns with 'user' and 'bot' messages
        
    Returns:
        Formatted string of the chat history
    """
    if not history:
        return ""
        
    lines = ["\n--- Chat History ---"]
    for turn in history:
        if "user" in turn:
            lines.append(f"You: {turn['user']}")
        if "bot" in turn:
            lines.append(f"Bot: {turn['bot']}")
    lines.append("-" * 20)
    return "\n".join(lines)

def main():
    """Run the interactive chat interface for MedBot."""
    try:
        # Initialize the conversation graph
        conversation = build_graph()
        
        # Configure session
        session_id = "medbot-session-1"  # Could be made configurable
        config = {"configurable": {"thread_id": session_id}}
        
        # Display welcome message
        print("=" * 50)
        print("MedBot - Medical Information Assistant")
        print("=" * 50)
        print("Type 'exit' or 'quit' to end the session.\n")
        
        # Main chat loop
        while True:
            try:
                # Get user input
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nEnding session. Goodbye!")
                    break
                
                # Check for exit conditions
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    print("\nEnding session. Goodbye!")
                    break
                
                # Process the query
                response = conversation.invoke(
                    {"query": user_input},
                    config=config
                )
                
                # Display the response
                bot_response = response.get("answer", "I'm sorry, I couldn't generate a response.")
                print(f"\nBot: {bot_response}")
                
                # Show conversation history if available
                history = response.get("history", [])
                if history:
                    print(format_chat_history(history))
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again or type 'exit' to quit.\n")
                
    except Exception as e:
        print(f"\nA fatal error occurred: {str(e)}")
        return 1  # Non-zero exit code on error
        
    return 0  # Success

if __name__ == "__main__":
    main()
