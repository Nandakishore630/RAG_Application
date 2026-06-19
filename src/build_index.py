from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import pickle
import os

# Load PDF
loader = PyPDFLoader(r"C:\Users\nanda\Desktop\financial_rag-assistant\data\apple-25.pdf")
documents = loader.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)

# FAISS
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings, dtype="float32"))

print("Vectors Stored:", index.ntotal)

# Create folder
os.makedirs("vector_store", exist_ok=True)

# Save FAISS index
faiss.write_index(
    index,
    "vector_store/apple_index.faiss"
)

# Save chunks
with open(
    "vector_store/chunks.pkl",
    "wb"
) as f:
    pickle.dump(chunks, f)

print("Index saved successfully!")