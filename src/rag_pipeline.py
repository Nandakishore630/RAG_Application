from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from langchain_ollama import ChatOllama

import faiss
import numpy as np

# Load PDF
loader = PyPDFLoader(r"C:\Users\nanda\Desktop\financial_rag-assistant\data\apple-25.pdf")
documents = loader.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Embeddings
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [chunk.page_content for chunk in chunks]

embeddings = embedding_model.encode(texts)

# FAISS
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings, dtype="float32"))

# Question
question = input("Ask a question: ")

# Query embedding
query_embedding = embedding_model.encode([question])

# Retrieve Top 3
distances, indices = index.search(
    np.array(query_embedding, dtype="float32"),
    k=3
)

context = "\n\n".join(
    [chunks[idx].page_content for idx in indices[0]]
)

# LLM
llm = ChatOllama(model="llama3.2")

prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)

print("\nANSWER:\n")
print(response.content)