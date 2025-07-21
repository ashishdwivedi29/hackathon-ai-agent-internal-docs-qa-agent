import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file (for GOOGLE_API_KEY)
load_dotenv()

# Define paths
# This should be the DIRECTORY where the index files will be stored.
FAISS_STORAGE_DIR = "data" 

def load_documents():
    """
    Loads documents from the 'docs' directory using appropriate loaders.
    Supports PDF, DOCX, and TXT/Markdown files.
    """
    documents = []
    print(f"Loading documents from 'docs' directory...")

    # Load PDF files
    pdf_loader = DirectoryLoader(
        "docs", # Use "docs" directly
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True 
    )
    documents.extend(pdf_loader.load())
    print(f"  - Loaded {len(documents)} PDF documents so far.")

    # Load DOCX files
    docx_loader = DirectoryLoader(
        "docs", # Use "docs" directly
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents.extend(docx_loader.load())
    print(f"  - Loaded {len(documents)} PDF/DOCX documents so far.")

    # Load TXT files
    txt_loader = DirectoryLoader(
        "docs", # Use "docs" directly
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents.extend(txt_loader.load())
    print(f"  - Loaded {len(documents)} PDF/DOCX/TXT documents so far.")
    
    # Load Markdown files
    md_loader = DirectoryLoader(
        "docs", # Use "docs" directly
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents.extend(md_loader.load())
    print(f"  - Loaded {len(documents)} PDF/DOCX/TXT/MD documents so far.")

    if not documents:
        print(f"No documents found in 'docs'. Please ensure your documents are in this folder.")
    else:
        print(f"Successfully loaded a total of {len(documents)} documents.")
    return documents

def split_documents(documents):
    """
    Splits loaded documents into smaller, manageable chunks.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    return chunks

def create_and_save_vector_store(chunks):
    """
    Generates embeddings for document chunks and saves them to a FAISS vector store.
    """
    if not chunks:
        print("No chunks to process. Skipping vector store creation.")
        return

    print("Generating embeddings and creating FAISS vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = FAISS.from_documents(chunks, embeddings)

    # Ensure the storage directory exists
    os.makedirs(FAISS_STORAGE_DIR, exist_ok=True) # THIS IS THE FIX: use FAISS_STORAGE_DIR directly

    # Save the FAISS index with a base name inside the specified directory
    vector_store.save_local(os.path.join(FAISS_STORAGE_DIR, "my_rag_index"))

    print(f"FAISS vector store successfully saved to '{FAISS_STORAGE_DIR}/my_rag_index'.")

def ingest_documents():
    """
    Main function to orchestrate the document ingestion process.
    """
    print("--- Starting Document Ingestion Process ---")
    documents = load_documents()

    if documents:
        chunks = split_documents(documents)
        create_and_save_vector_store(chunks)
    else:
        print("No documents were loaded, skipping chunking and vector store creation.")
        print("Please ensure you have documents in the 'docs/' folder.")

    print("--- Document Ingestion Process Complete ---")

if __name__ == "__main__":
    ingest_documents()