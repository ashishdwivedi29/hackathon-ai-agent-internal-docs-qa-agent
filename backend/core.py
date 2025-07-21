import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader # For dynamic loading if implemented
from langchain.text_splitter import RecursiveCharacterTextSplitter # For dynamic loading

# Load environment variables
load_dotenv()

# This should be the DIRECTORY where the index files are stored.
FAISS_STORAGE_DIR = "data" 

def load_vector_store():
    """
    Loads the FAISS vector store from the local path.
    Returns None if the index files do not exist.
    """
    # We check for the .faiss file using the full expected path
    full_index_path = os.path.join(FAISS_STORAGE_DIR, "my_rag_index")
    # Corrected check: ensure it looks inside the directory for the specific index file
    if not os.path.exists(os.path.join(full_index_path, "index.faiss")): 
        print(f"FAISS index files not found at {full_index_path}. Please run ingest.py first or ensure path is correct.")
        return None
    
    print(f"Loading FAISS vector store from {full_index_path}...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # allow_dangerous_deserialization=True is needed for FAISS.load_local to deserialize the index
    vector_store = FAISS.load_local(full_index_path, embeddings, allow_dangerous_deserialization=True)
    print("FAISS vector store loaded successfully.")
    return vector_store

def get_rag_chain(vector_store):
    """
    Creates and returns a Retrieval-Augmented Generation (RAG) chain.
    """
    if vector_store is None:
        raise ValueError("Vector store must be loaded before creating RAG chain.")

    print("Setting up RAG chain...")
    # Using gemini-1.5-flash for faster responses and better free-tier limits
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3) 
    
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for answering questions about internal documents.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide the answer concisely and accurately based ONLY on the provided context.
If the context contains conflicting information, state that.

Context: {context}

Question: {input}
""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("RAG chain created.")
    return retrieval_chain

# --- NEW FUNCTIONALITY: Automated FAQ Generation ---
def generate_faqs(document_content: str) -> str:
    """
    Generates FAQs from the provided document content using Gemini.
    """
    if not document_content:
        return "No document content provided to generate FAQs."

    print("Generating FAQs...")
    # Using gemini-1.5-flash for FAQ generation, with a slightly higher temperature
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7) 

    # Craft the prompt for FAQ generation
    faq_prompt = ChatPromptTemplate.from_template(f"""
You are an expert at creating concise and helpful Frequently Asked Questions (FAQs) from internal company documents.
Your goal is to extract the most important information and present it as Q&A pairs.

Based ONLY on the document content provided below, generate 5 to 8 relevant and distinct FAQs and their answers.
Ensure the answers are direct and come directly from the provided text.
Format your output strictly as follows:

**Q:** [Question 1]
**A:** [Answer 1]

**Q:** [Question 2]
**A:** [Answer 2]

... (continue for 5-8 Q&A pairs)

Document Content:
---
{{document_content}}
---
""")

    # Invoke the LLM with the document content
    try:
        chain = faq_prompt | llm
        response = chain.invoke({"document_content": document_content})
        print("FAQs generated successfully.")
        return response.content
    except Exception as e:
        print(f"Error generating FAQs: {e}")
        return f"An error occurred while generating FAQs: {e}"

# --- Dynamic Document Addition Function ---
def add_uploaded_documents_to_faiss(uploaded_file_paths, current_vector_store):
    """
    Loads, splits, and adds new documents to an existing FAISS vector store.
    Returns the updated vector store and the combined text content of new documents.
    """
    if not uploaded_file_paths:
        print("No new files to add.")
        return current_vector_store, "" # Return empty string for content

    print(f"Processing {len(uploaded_file_paths)} new uploaded documents...")
    new_documents = []
    for file_path in uploaded_file_paths:
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith(".txt") or file_path.endswith(".md"):
                loader = TextLoader(file_path)
            else:
                print(f"Unsupported file type: {file_path}. Skipping.")
                continue
            
            new_documents.extend(loader.load())
            print(f"  - Loaded {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Extract and combine the raw text content from the newly loaded documents
    combined_new_docs_content = ""
    for doc in new_documents:
        combined_new_docs_content += doc.page_content + "\n\n" # Combine content of all new docs

    if not new_documents:
        print("No valid new documents loaded from upload.")
        return current_vector_store, "" # Return empty string for content

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    new_chunks = text_splitter.split_documents(new_documents)
    print(f"Created {len(new_chunks)} chunks from new documents.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if current_vector_store:
        print("Adding new chunks to existing FAISS vector store...")
        current_vector_store.add_documents(new_chunks, embeddings=embeddings)
        print("New chunks added.")
    else:
        print("No existing vector store found. Creating a new one with new documents.")
        current_vector_store = FAISS.from_documents(new_chunks, embeddings)

    # Save the updated (or new) vector store
    os.makedirs(FAISS_STORAGE_DIR, exist_ok=True) # Ensure data directory exists
    current_vector_store.save_local(os.path.join(FAISS_STORAGE_DIR, "my_rag_index")) # Save with a base name

    print(f"Updated FAISS vector store saved to '{FAISS_STORAGE_DIR}/my_rag_index'.")

    return current_vector_store, combined_new_docs_content # Return both vector store and content




# import os
# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader # For dynamic loading if implemented
# from langchain.text_splitter import RecursiveCharacterTextSplitter # For dynamic loading

# # Load environment variables
# load_dotenv()

# # This should be the DIRECTORY where the index files are stored.
# FAISS_STORAGE_DIR = "data" 

# def load_vector_store():
#     """
#     Loads the FAISS vector store from the local path.
#     Returns None if the index files do not exist.
#     """
#     # We check for the .faiss file using the full expected path
#     full_index_path = os.path.join(FAISS_STORAGE_DIR, "my_rag_index")
#     if not os.path.exists(os.path.join(full_index_path, "index.faiss")): # CORRECTED CHECK:
#         print(f"FAISS index files not found at {full_index_path}. Please run ingest.py first or ensure path is correct.")
#         return None
    
#     print(f"Loading FAISS vector store from {full_index_path}...")
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     vector_store = FAISS.load_local(full_index_path, embeddings, allow_dangerous_deserialization=True)
#     print("FAISS vector store loaded successfully.")
#     return vector_store

# def get_rag_chain(vector_store):
#     """
#     Creates and returns a Retrieval-Augmented Generation (RAG) chain.
#     """
#     if vector_store is None:
#         raise ValueError("Vector store must be loaded before creating RAG chain.")

#     print("Setting up RAG chain...")
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3) 
    
#     retriever = vector_store.as_retriever()

#     prompt = ChatPromptTemplate.from_template("""
# You are an AI assistant for answering questions about internal documents.
# Use the following context to answer the question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Provide the answer concisely and accurately based ONLY on the provided context.
# If the context contains conflicting information, state that.

# Context: {context}

# Question: {input}
# """)

#     document_chain = create_stuff_documents_chain(llm, prompt)

#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
#     print("RAG chain created.")
#     return retrieval_chain

# # --- Dynamic Document Addition Function ---
# def add_uploaded_documents_to_faiss(uploaded_file_paths, current_vector_store):
#     """
#     Loads, splits, and adds new documents to an existing FAISS vector store.
#     Returns the updated vector store.
#     """
#     if not uploaded_file_paths:
#         print("No new files to add.")
#         return current_vector_store

#     print(f"Processing {len(uploaded_file_paths)} new uploaded documents...")
#     new_documents = []
#     for file_path in uploaded_file_paths:
#         try:
#             if file_path.endswith(".pdf"):
#                 loader = PyPDFLoader(file_path)
#             elif file_path.endswith(".docx"):
#                 loader = Docx2txtLoader(file_path)
#             elif file_path.endswith(".txt") or file_path.endswith(".md"):
#                 loader = TextLoader(file_path)
#             else:
#                 print(f"Unsupported file type: {file_path}. Skipping.")
#                 continue
            
#             new_documents.extend(loader.load())
#             print(f"  - Loaded {os.path.basename(file_path)}")
#         except Exception as e:
#             print(f"Error loading {file_path}: {e}")

#     if not new_documents:
#         print("No valid new documents loaded from upload.")
#         return current_vector_store

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         add_start_index=True,
#     )
#     new_chunks = text_splitter.split_documents(new_documents)
#     print(f"Created {len(new_chunks)} chunks from new documents.")

#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     if current_vector_store:
#         print("Adding new chunks to existing FAISS vector store...")
#         current_vector_store.add_documents(new_chunks, embeddings=embeddings)
#         print("New chunks added.")
#     else:
#         print("No existing vector store found. Creating a new one with new documents.")
#         current_vector_store = FAISS.from_documents(new_chunks, embeddings)

#     # Save the updated (or new) vector store
#     os.makedirs(FAISS_STORAGE_DIR, exist_ok=True) # Ensure data directory exists
#     current_vector_store.save_local(os.path.join(FAISS_STORAGE_DIR, "my_rag_index")) # Save with a base name

#     print(f"Updated FAISS vector store saved to '{FAISS_STORAGE_DIR}/my_rag_index'.")

#     return current_vector_store