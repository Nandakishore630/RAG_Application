from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load PDF
loader = PyPDFLoader(r"C:\Users\nanda\Desktop\financial_rag-assistant\data\apple-25.pdf")
documents = loader.load()

# Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)

# Build FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings, dtype="float32"))

# Question
query = input("Ask a question: ")

# Convert question into embedding
query_embedding = model.encode([query])

# Search Top 3 chunks
distances, indices = index.search(
    np.array(query_embedding, dtype="float32"),
    k=3
)

print("\nTop Relevant Chunks:\n")

for idx in indices[0]:
    print("=" * 80)
    print(chunks[idx].page_content[:1000])
    print("\n")