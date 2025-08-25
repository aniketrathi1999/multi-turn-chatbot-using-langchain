import pandas as pd
from langchain.docstore.document import Document
from vector_store import get_vector_store
from dotenv import load_dotenv
import os
load_dotenv()

def ingest_csv(path=os.getenv("CSV_PATH")):
    """Read and process a CSV file containing medical data.
    
    Args:
        path: Path to the CSV file. If not provided, uses CSV_PATH from environment.
        
    Returns:
        List of Document objects ready for vector storage
        
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
        required_columns = {"Medicine Name"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Process each row into a document
        docs = []
        for _, row in df.iterrows():
            # Skip rows without a medicine name
            if pd.isna(row.get("Medicine Name")):
                continue
                
            # Build document content
            medicine_name = str(row["Medicine Name"]).strip()
            uses = str(row.get("Uses", "")).strip()
            side_effects = str(row.get("Side_effects", "")).strip()
            
            # Create document with metadata
            doc = Document(
                page_content=f"Medicine: {medicine_name} | Uses: {uses} | Side Effects: {side_effects}",
                metadata={
                    "name": medicine_name,
                    "source": os.path.basename(path),
                    "has_uses": bool(uses),
                    "has_side_effects": bool(side_effects)
                }
            )
            docs.append(doc)
            
        return docs
        
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

def main():
    """Main function to handle document ingestion process."""
    try:
        # Initialize vector store
        vector_store = get_vector_store()
        
        # Process and validate documents
        docs = ingest_csv()
        if not docs:
            print("No valid documents found to ingest.")
            return
            
        # Ingest documents into vector store
        doc_ids = vector_store.add_documents(docs)
        
        print(f"Successfully ingested {len(doc_ids)} documents into Pinecone.")
        
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        return 1  # Return non-zero exit code on error
        
    return 0  # Return success

if __name__ == "__main__":
    main()
