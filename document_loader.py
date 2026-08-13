import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdfs(pdf_paths):
    """
    Extract text from a list of PDF file paths.
    Returns a list of LangChain Document objects with page text and metadata.
    """
    documents = []
    
    for path in pdf_paths:
        try:
            reader = PdfReader(path)
            file_name = os.path.basename(path)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                # Skip empty pages safely
                if text and text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "page": page_num + 1 # 1-indexed page numbers for human readability
                        }
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    return documents

def get_text_chunks(documents, chunk_size=800, chunk_overlap=120):
    """
    Split the extracted documents into smaller chunks.
    Retains metadata (document name, page number) in each chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
