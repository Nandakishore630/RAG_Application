from urllib import response
import streamlit as st
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

# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Load LLM
llm = ChatOllama(
    model="llama3.2"
)

def ask_question(question):

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
Answer only from the context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    sources = []

    for idx in indices[0]:
        page = chunks[idx].metadata.get(
            "page",
            "Unknown"
        )

        sources.append({
    "page": page + 1,
    "text": chunks[idx].page_content[:300]})

    return response.content, sources


# Streamlit UI

st.title("Financial Intelligence RAG Assistant")

question = st.text_input(
    "Ask a question about financial reports"
)
st.sidebar.title("Document Information")
st.sidebar.write(f"Chunks: {len(chunks)}")
st.sidebar.write(f"Vectors: {index.ntotal}")
if question:
    answer, sources = ask_question(question)
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for source in sources:
        st.write(source)


