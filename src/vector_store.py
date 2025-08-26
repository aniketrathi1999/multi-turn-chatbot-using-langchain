import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load env vars
load_dotenv(".env.local")

INDEX_NAME = os.getenv("MEDICINES_INDEX_NAME")

def get_pinecone_index():
    """Get or create a Pinecone index for vector storage.
    
    Returns:
        A Pinecone index instance ready for vector operations
        
    Raises:
        ValueError: If PINECONE_API_KEY is not set in environment variables
        RuntimeError: If there's an issue with Pinecone setup or index creation
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
        
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # List all indexes to verify connection
        try:
            existing_indexes = pc.list_indexes()
            index_names = [index.name for index in existing_indexes]  # Updated for new Pinecone client
        except Exception as e:
            raise RuntimeError(f"Failed to list Pinecone indexes. Please check your API key and environment. Error: {str(e)}")
        
        # Create index if it doesn't exist
        if INDEX_NAME not in index_names:
            try:
                print(f"Creating new Pinecone index: {INDEX_NAME}")
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=1536,  # Matches text-embedding-3-small model
                    metric="cosine",  # Best for semantic similarity
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Using us-east-1 as default
                    )
                )
                
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                import time
                time.sleep(30)  # Increased wait time for index initialization
                
            except Exception as e:
                raise RuntimeError(f"Failed to create Pinecone index. Error: {str(e)}")
        
        # Return the index
        return pc.Index(INDEX_NAME)
        
    except Exception as e:
        raise RuntimeError(f"Pinecone initialization failed: {str(e)}")

def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """Create a text splitter with specified chunk size and overlap.
    
    Args:
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        Configured RecursiveCharacterTextSplitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

def get_vector_store(chunk_size: int = 1000, chunk_overlap: int = 200):
    """Initialize and return a vector store for document operations.
    
    Args:
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        A tuple of (PineconeVectorStore, text_splitter) configured for document operations
        
    Raises:
        RuntimeError: If there's an issue initializing the vector store or creating the index
    """
    try:
        # Initialize embeddings and text splitter
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        text_splitter = get_text_splitter(chunk_size, chunk_overlap)
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Check if index exists, create if it doesn't
        try:
            # First try to get the index to see if it exists
            index = pc.Index(INDEX_NAME)
            index.describe_index_stats()  # This will raise an exception if index doesn't exist
        except Exception:
            # If index doesn't exist, create it
            print(f"Index '{INDEX_NAME}' not found. Creating a new index...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Waiting for index to be ready...")
            import time
            time.sleep(30)  # Wait for index to be ready
        
        # Initialize the vector store
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key="text"
        )
        
        return vector_store, text_splitter
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize vector store: {str(e)}")
