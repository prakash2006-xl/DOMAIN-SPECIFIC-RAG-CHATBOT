import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# Fix for SSL certificate path issue [Errno 2] No such file or directory
# We downloaded a clean cacert.pem and force httpx to use it exclusively.
cert_path = os.path.join(os.path.dirname(__file__), "cacert.pem")
os.environ["SSL_CERT_FILE"] = cert_path
os.environ["REQUESTS_CA_BUNDLE"] = cert_path
os.environ.pop("SSL_CERT_DIR", None)

# Load environment variables (override to ensure it catches latest changes to .env)
load_dotenv(override=True)

from document_loader import extract_text_from_pdfs, get_text_chunks
from vector_store import create_and_save_vector_store, load_vector_store
from rag_pipeline import retrieve_and_answer

# Setup Streamlit page configuration
st.set_page_config(page_title="Domain-Specific RAG Chatbot", layout="wide")

# Constants
UPLOAD_DIR = "documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
VECTOR_STORE_DIR = os.path.join("vector_store", "saved_index")

def save_uploaded_files(uploaded_files):
    """Save uploaded files to the local directory"""
    saved_paths = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)
    return saved_paths

def clear_documents():
    """Clear uploaded documents and the vector store"""
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error deleting file {file_path}: {e}")
            
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
        
    st.session_state.vector_store = None
    st.success("Documents and Vector Store cleared successfully.")

def process_documents(uploaded_files):
    """Extract, chunk, and embed documents"""
    if not uploaded_files:
        st.warning("Please upload PDF documents first.")
        return
        
    # Check file sizes
    MAX_FILE_SIZE_MB = 10
    for file in uploaded_files:
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File {file.name} exceeds the {MAX_FILE_SIZE_MB}MB size limit. Please upload smaller files.")
            return
        
    with st.spinner("Processing documents... This may take a moment."):
        saved_paths = save_uploaded_files(uploaded_files)
        if not saved_paths:
            st.error("No valid PDF files were saved.")
            return
            
        st.info("Extracting text from PDFs...")
        documents = extract_text_from_pdfs(saved_paths)
        
        if not documents:
            st.error("No text could be extracted from the uploaded PDFs. They might be empty or image-only.")
            return
            
        st.info("Splitting text into chunks...")
        chunks = get_text_chunks(documents)
        
        st.info("Creating embeddings and saving vector store...")
        vector_store = create_and_save_vector_store(chunks)
        st.session_state.vector_store = vector_store
        
        st.success("Documents processed successfully! You can now ask questions.")

def main():
    st.header("Domain-Specific RAG Chatbot")
    st.subheader("Ask questions from your uploaded PDF documents")
    st.info("⚠️ **Disclaimer:** AI-generated answers may not always be automatically correct. Please verify high-stakes information independently.")


    # Initialize session state for chat history and vector store
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store()
        
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Document Manager")
        st.warning("🔒 Do not upload confidential documents without permission.")
        uploaded_files = st.file_uploader(
            "Upload your PDF documents (Max 10MB per file)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write("Uploaded Files:")
            for file in uploaded_files:
                st.write(f"- {file.name}")
        
        if st.button("Process Documents", type="primary"):
            process_documents(uploaded_files)
            
        if st.button("Clear Documents"):
            clear_documents()
            
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- MAIN CHAT AREA ---
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**{i+1}. {source.metadata.get('source', 'Unknown')}** — Page {source.metadata.get('page', 'Unknown')}")
                        st.caption(f'"{source.page_content[:200]}..."')
            if message["role"] == "assistant":
                st.feedback("thumbs", key=f"hist_fb_{st.session_state.messages.index(message)}")

    # Chat Input
    if question := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Check preconditions
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            st.error("GROQ_API_KEY is not set in the .env file. Please add your Groq API key to the .env file to continue.")
            return
            
        if st.session_state.vector_store is None:
            # Try to load it from disk one more time just in case session state dropped it
            loaded_store = load_vector_store()
            if loaded_store is not None:
                st.session_state.vector_store = loaded_store
            else:
                st.warning("Please upload and process documents before asking questions.")
                return

        # Format chat history
        chat_history = ""
        for msg in st.session_state.messages[:-1]:  # Exclude current question
            chat_history += f"{msg['role'].capitalize()}: {msg['content']}\n"
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching for answers..."):
                try:
                    answer, source_docs = retrieve_and_answer(
                        question=question,
                        chat_history=chat_history,
                        vector_store=st.session_state.vector_store,
                        api_key=api_key
                    )
                    
                    st.markdown(answer)
                    
                    # Display sources if answer was not a refusal and if there are sources
                    if "I could not find this information" not in answer and source_docs:
                        with st.expander("Sources"):
                            for i, source in enumerate(source_docs):
                                st.markdown(f"**{i+1}. {source.metadata.get('source', 'Unknown')}** — Page {source.metadata.get('page', 'Unknown')}")
                                st.caption(f'"{source.page_content[:200]}..."')
                                
                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": source_docs if "I could not find this information" not in answer else []
                    })
                    
                    st.feedback("thumbs", key=f"curr_fb_{len(st.session_state.messages)}")
                except Exception as e:
                    import traceback
                    st.error(f"An error occurred while generating the answer: {e}")
                    st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
