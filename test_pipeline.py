import os
import sys

# Ensure paths are correct
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_loader import extract_text_from_pdfs, get_text_chunks
from vector_store import create_and_save_vector_store, load_vector_store

def run_tests():
    print("--- Testing RAG Pipeline Local Components ---")
    
    # 1. Test Document Loading
    pdf_path = os.path.join("documents", "sample.pdf")
    if not os.path.exists(pdf_path):
        print(f"FAILED: Sample PDF not found at {pdf_path}")
        return
        
    print(f"\n1. Extracting text from {pdf_path}...")
    documents = extract_text_from_pdfs([pdf_path])
    if not documents:
        print("FAILED: No text extracted. Is the PDF empty?")
        return
    print(f"SUCCESS: Extracted {len(documents)} pages.")
    for doc in documents[:2]:
        print(f" - Page {doc.metadata.get('page')}: {len(doc.page_content)} characters")
        
    # 2. Test Chunking
    print("\n2. Chunking text...")
    chunks = get_text_chunks(documents)
    if not chunks:
        print("FAILED: No chunks created.")
        return
    print(f"SUCCESS: Created {len(chunks)} chunks.")
    print(f" - First chunk preview: {chunks[0].page_content[:50]}...")
    
    # 3. Test Vector Store Creation (Embeddings + FAISS)
    print("\n3. Creating Embeddings and Vector Store (FAISS)...")
    try:
        vector_store = create_and_save_vector_store(chunks)
        print("SUCCESS: Vector store created and saved successfully.")
    except Exception as e:
        print(f"FAILED: Error creating vector store: {e}")
        return
        
    # 4. Test Vector Store Loading and Similarity Search
    print("\n4. Loading Vector Store and performing Similarity Search...")
    try:
        loaded_store = load_vector_store()
        if loaded_store is None:
            print("FAILED: Could not load vector store.")
            return
            
        test_query = "What is the project objective?"
        print(f" - Query: '{test_query}'")
        retriever = loaded_store.as_retriever(search_kwargs={"k": 2})
        results = retriever.invoke(test_query)
        print(f"SUCCESS: Retrieved {len(results)} chunks.")
        for i, res in enumerate(results):
            print(f"   Result {i+1} (Page {res.metadata.get('page')}): {res.page_content[:100]}...")
    except Exception as e:
        print(f"FAILED: Error in similarity search: {e}")
        return

    print("\n--- All local tests passed! ---")
    print("The only remaining part is the Groq LLM which requires an API key.")

if __name__ == "__main__":
    run_tests()
