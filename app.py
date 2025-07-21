import streamlit as st
import os
import tempfile
# Removed: import yaml, from yaml.loader import SafeLoader, import streamlit_authenticator as stauth

# Ensure these imports are correct and match your backend/core.py
from backend.core import get_rag_chain, load_vector_store, add_uploaded_documents_to_faiss, generate_faqs
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Internal Docs Q&A Agent", page_icon="📚", layout="wide")
st.title("📚 Internal Docs Q&A Agent")

# --- NO USER AUTHENTICATION FOR NOW ---
# The entire app content runs directly without a login check

# --- Sidebar for Document Upload and FAQ Generation ---
with st.sidebar:
    st.header("Upload New Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT/MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader" 
    )

    # Session state to store combined content of last upload for FAQ generation
    if "last_uploaded_raw_content" not in st.session_state:
        st.session_state.last_uploaded_raw_content = ""

    if uploaded_files:
        if st.button("Process & Index Uploaded Documents", key="process_button"): # Clearer button text
            with st.spinner("Processing documents for indexing & FAQ generation..."):
                temp_dir = tempfile.mkdtemp() 
                temp_file_paths = []
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_file_paths.append(file_path)

                try:
                    # Call add_uploaded_documents_to_faiss which now returns content
                    updated_vector_store, combined_uploaded_content = add_uploaded_documents_to_faiss(
                        temp_file_paths, st.session_state.vector_store
                    )
                    st.session_state.vector_store = updated_vector_store
                    st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store) 
                    st.success("Documents uploaded and indexed successfully!")
                    st.session_state.messages.append(AIMessage(content="New documents have been indexed and are ready for questions."))
                    
                    # Store the combined content for potential FAQ generation
                    st.session_state.last_uploaded_raw_content = combined_uploaded_content
                    
                except Exception as e:
                    st.error(f"Error processing uploaded documents: {e}")
                    st.session_state.last_uploaded_raw_content = "" # Clear content on error
                finally:
                    import shutil
                    shutil.rmtree(temp_dir)
    
    # New: FAQ Generation Button (appears after documents are processed/content is available)
    if st.session_state.last_uploaded_raw_content:
        st.markdown("---")
        if st.button("Generate FAQs for Last Uploaded Document(s)", key="generate_faq_button"):
            if st.session_state.last_uploaded_raw_content.strip(): # Check if content is not empty
                with st.spinner("Generating FAQs... This may take a moment."):
                    faqs_output = generate_faqs(st.session_state.last_uploaded_raw_content)
                    st.subheader("Generated FAQs:")
                    st.markdown(faqs_output) # Display raw markdown from LLM
                    # Optionally, add these FAQs to the chat history
                    st.session_state.messages.append(AIMessage(content="Here are some generated FAQs:"))
                    st.session_state.messages.append(AIMessage(content=faqs_output))
            else:
                st.warning("No extractable content from the last upload to generate FAQs from.")


    st.markdown("---")
    st.info("This agent answers questions based on documents loaded into its knowledge base.")
    st.info("Make sure you've run `python backend/ingest.py` at least once to build the initial knowledge base from the `docs/` folder.")
    
# --- Main Chat Interface ---

# Initialize chat history and RAG components in session state if not present
# Simplified history key (no username specific history for now)
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! I am your Internal Docs Q&A Agent. How can I help you today?")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Load or create RAG chain and vector store when the app starts or if not yet loaded
if st.session_state.vector_store is None:
    with st.spinner("Loading knowledge base..."):
        try:
            st.session_state.vector_store = load_vector_store()
            if st.session_state.vector_store:
                st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store)
            else:
                st.warning("No knowledge base found. Please run `python backend/ingest.py` to create one from your `docs/` folder.")
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            st.session_state.vector_store = None 

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.rag_chain:
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": prompt})
                    ai_response = response.get("answer", "I couldn't find an answer in the documents.")
                    
                    st.markdown(ai_response)

                    if "source_documents" in response and response["source_documents"]:
                        st.markdown("\n\n---") 
                        st.markdown("**Sources:**")
                        for i, doc in enumerate(response["source_documents"]):
                            source_path = doc.metadata.get('source', 'N/A')
                            page_number = doc.metadata.get('page', 'N/A') 
                            
                            display_source = os.path.basename(source_path)
                            
                            source_info = f"- **{display_source}**"
                            if page_number != 'N/A':
                                source_info += f" (Page: {page_number})"
                            
                            st.markdown(source_info)

                except Exception as e:
                    st.error(f"An error occurred while getting the AI response: {e}")
                    ai_response = "Sorry, I'm having trouble answering that question right now."
    else:
        ai_response = "The knowledge base is not loaded. Please ensure documents are processed via `python backend/ingest.py` or uploaded and processed via the sidebar."
        with st.chat_message("ai"):
            st.markdown(ai_response)
            
    st.session_state.messages.append(AIMessage(content=ai_response))

