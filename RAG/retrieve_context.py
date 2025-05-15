from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def retrieve_documents(query: str, category: str):
    # Load embedding model
    embedding_model_name = "msmarco-distilbert-base-tas-b"
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Load FAISS vector store
    vectorstore = FAISS.load_local(
        "vector_store",
        embedding_function,
        allow_dangerous_deserialization=True 
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever()

    # Get results with category filter
    results = retriever.invoke(
        query,
        filters={"category": category}
    )

    return results