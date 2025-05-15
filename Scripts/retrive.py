from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Step 1: Load the same embedding model you used during saving
embedding_model_name = "msmarco-distilbert-base-tas-b"
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Step 2: Load FAISS vector store from disk
vectorstore = FAISS.load_local(
    "faiss_store_rich",
    embedding_function,
    allow_dangerous_deserialization=True 
)
# Step 3: Wrap as a retriever
retriever = vectorstore.as_retriever()

# Step 4: Apply metadata filter (e.g., category == 'License_Agreements')
query = "What specific maintenance services is Netzee obligated to provide, and during what hours is customer support available?"
results = retriever.invoke(
    query,
    filters={"category": "Maintenance"}  # only fetch matching category
)

# Step 5: Show results
for doc in results:
    print("📄", doc.metadata["doc_name"])
    print("📌 Category:", doc.metadata["category"])
    print("📝 Preview:", doc.page_content)
    print("=" * 80)