from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import pickle
import os

# Load all documents
documents = []

data_folder = r"C:\Users\nanda\Desktop\financial_rag-assistant\data"

for file in os.listdir(data_folder):

    file_path = os.path.join(data_folder, file)

    try:

        if file.endswith(".pdf"):

            loader = PyPDFLoader(file_path)
            docs = loader.load()

        elif file.endswith(".docx"):

            loader = Docx2txtLoader(file_path)
            docs = loader.load()

        else:
            continue

        for doc in docs:
            doc.metadata["source_file"] = file

        documents.extend(docs)

        print(f"Loaded: {file}")

    except Exception as e:

        print(f"Error loading {file}: {e}")

print(f"\nTotal Documents Loaded: {len(documents)}")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

# Embeddings
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)

# FAISS Index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(
        embeddings,
        dtype="float32"
    )
)

print(f"Vectors Stored: {index.ntotal}")

# Create vector store folder
os.makedirs(
    "vector_store",
    exist_ok=True
)

# Save FAISS index
faiss.write_index(
    index,
    "vector_store/document_index.faiss"
)

# Save chunks
with open(
    "vector_store/chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )

print("\nIndex saved successfully!")