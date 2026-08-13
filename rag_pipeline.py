from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from prompt import rag_prompt

def format_docs(docs):
    """
    Format the retrieved documents into a single string to be passed as context.
    """
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        content = doc.page_content
        formatted_docs.append(f"Source: {source}, Page: {page}\nContent:\n{content}")
    return "\n\n".join(formatted_docs)

def get_rag_chain(retriever, api_key):
    """
    Construct the RAG pipeline.
    """
    # Initialize the LLM (Groq)
    # Using Llama-3.1-8b as a fast, reliable model on Groq
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant",
        api_key=api_key
    )
    
    # Build the chain
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs, 
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def retrieve_and_answer(question, chat_history, vector_store, api_key):
    """
    Given a question, chat history, and vector store, retrieve relevant chunks and generate answer.
    Returns the answer and the source documents used.
    """
    # Retrieve top 3-5 relevant chunks (k=4)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Get the chain
    chain = get_rag_chain(retriever, api_key)
    
    # Invoke the chain
    answer = chain.invoke({"question": question, "chat_history": chat_history})
    
    # Fetch the documents directly to return them for source display
    source_docs = retriever.invoke(question)
    
    return answer, source_docs
