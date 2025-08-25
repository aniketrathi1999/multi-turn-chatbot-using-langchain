import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load env vars
load_dotenv(".env.local")

INDEX_NAME = "medicines-index"

def get_pinecone_index():
    """Get or create a Pinecone index for vector storage.
    
    Returns:
        A Pinecone index instance ready for vector operations
        
    Raises:
        ValueError: If PINECONE_API_KEY is not set in environment variables
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
        
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    
    # Create index if it doesn't exist
    if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Matches text-embedding-3-small model
            metric="cosine",  # Best for semantic similarity
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Choose region closest to your users
            )
        )
        
        # Brief pause to ensure index is ready
        import time
        time.sleep(10)
        
    return pc.Index(INDEX_NAME)

def get_vector_store():
    """Initialize and return a vector store for document operations.
    
    Returns:
        A PineconeVectorStore instance configured for document operations
        
    Raises:
        RuntimeError: If there's an issue initializing the vector store
    """
    try:
        # Initialize embeddings model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Get or create the Pinecone index
        index = get_pinecone_index()
        
        # Initialize the vector store
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"  # Field name for document content
        )
        
        return vector_store
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize vector store: {str(e)}")
