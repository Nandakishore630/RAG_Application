import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama

# Load FAISS index
index = faiss.read_index(
    "vector_store/apple_index.faiss"
)

# Load chunks
with open(
    "vector_store/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)

# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# LLM
llm = ChatOllama(
    model="llama3.2"
)

while True:

    question = input(
        "\nAsk a question (or type exit): "
    )

    if question.lower() == "exit":
        break

    query_embedding = embedding_model.encode(
        [question]
    )

    distances, indices = index.search(
        np.array(
            query_embedding,
            dtype="float32"
        ),
        k=3
    )

    context = "\n\n".join(
        [
            chunks[idx].page_content
            for idx in indices[0]
        ]
    )

    prompt = f"""
You are a financial analyst.

Answer only from the context.

If the answer is not present,
say you could not find it.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nANSWER:\n")
    print(response.content)