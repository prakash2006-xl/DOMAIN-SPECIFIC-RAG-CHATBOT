import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_STORE_PATH = os.path.join("vector_store", "saved_index")

def get_embedding_model():
    """
    Initialize the all-MiniLM-L6-v2 embedding model using HuggingFace.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    return embeddings

def create_and_save_vector_store(text_chunks):
    """
    Convert text chunks to embeddings and store in FAISS.
    Saves the FAISS index locally.
    """
    embeddings = get_embedding_model()
    
    # Create the vector store
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    
    # Ensure directory exists before saving
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    
    # Save the vector store locally
    vector_store.save_local(VECTOR_STORE_PATH)
    
    return vector_store

def load_vector_store():
    """
    Load the FAISS vector store from local storage.
    """
    if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        return None
        
    embeddings = get_embedding_model()
    # Allow dangerous deserialization because we created the file locally
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return vector_store
