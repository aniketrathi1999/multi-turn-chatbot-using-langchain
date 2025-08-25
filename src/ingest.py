import pandas as pd
from langchain.docstore.document import Document
from vector_store import get_vector_store
from dotenv import load_dotenv
import os
load_dotenv()

def ingest_csv(path=os.getenv("CSV_PATH"), chunk_size: int = 1000, chunk_overlap: int = 200):
    """Read and process a CSV file containing medical data.
    
    Args:
        path: Path to the CSV file. If not provided, uses CSV_PATH from environment.
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of Document objects ready for vector storage, split into chunks with overlap
        
    Raises:
        FileNotFoundError: If the specified CSV file doesn't exist
        ValueError: If required columns are missing or file is empty
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found at: {path}")
    
    try:
        # Read and validate CSV
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV file is empty")
            
        # Check for required columns
        required_columns = {"product_name", "medicine_desc", "side_effects", "product_price"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Process each row into a document
        docs = []
        for _, row in df.iterrows():
            # Skip rows without a product_name
            if pd.isna(row.get("product_name")):
                continue
                
            # Build document content
            medicine_name = str(row["product_name"]).strip()
            uses = str(row.get("medicine_desc", "")).strip()
            side_effects = str(row.get("side_effects", "")).strip()
            product_price = str(row.get("product_price", "")).strip()

            # Create document with metadata
            doc_text = f"Medicine: {medicine_name}\n\nUses: {uses}\n\nSide Effects: {side_effects}\n\nProduct Price: {product_price}"
            
            metadata = {
                "name": medicine_name,
                "source": os.path.basename(path),
                "has_uses": bool(uses),
                "has_side_effects": bool(side_effects)
            }
            
            # Create base document
            doc = Document(
                page_content=doc_text,
                metadata=metadata
            )
            docs.append(doc)
            
        return docs
        
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

def main():
    """Main function to handle document ingestion process with chunking."""
    try:
        # Configuration
        chunk_size = 1000  # characters
        chunk_overlap = 200  # characters
        
        # Initialize vector store and text splitter
        vector_store, text_splitter = get_vector_store(chunk_size, chunk_overlap)
        
        # Process and validate documents
        docs = ingest_csv(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not docs:
            print("No valid documents found to ingest.")
            return 1
            
        # Split documents into chunks with overlap
        chunked_docs = []
        doc_index = 0 
        for doc in docs:
            # Split the document text into chunks
            chunks = text_splitter.split_text(doc.page_content)
            
            # Create new documents for each chunk with metadata
            for i, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk"] = f"{i+1}/{len(chunks)}"
                
                chunked_doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                chunked_docs.append(chunked_doc)
            doc_index += 1
            print("Reading index: ", doc_index)
        # Ingest chunks into vector store
        doc_ids = vector_store.add_documents(chunked_docs)
        
        print(f"Successfully ingested {len(doc_ids)} document chunks into Pinecone.")
        print(f"Original documents: {len(docs)}")
        print(f"Total chunks created: {len(chunked_docs)}")
        
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1  # Return non-zero exit code on error
        
    return 0  # Return success

if __name__ == "__main__":
    main()
