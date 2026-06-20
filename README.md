# Financial Intelligence RAG Assistant

## Overview

Financial Intelligence RAG Assistant is a Retrieval-Augmented Generation (RAG) application that enables users to ask questions about financial reports and receive context-aware answers.

The system processes financial documents, converts them into embeddings, stores them in a FAISS vector database, retrieves relevant information using semantic search, and generates answers using a local Large Language Model (LLM) powered by Ollama.

---

## Features

* Multi-document support (PDF and DOCX)
* Document chunking using LangChain
* Semantic embeddings using Sentence Transformers
* Vector similarity search using FAISS
* Local LLM inference using Ollama and Llama 3.2
* Retrieval-Augmented Generation (RAG)
* Streamlit-based user interface
* Source-aware document retrieval
* Scalable architecture for enterprise document search

---

## Architecture

```text
Documents (PDF/DOCX)
          │
          ▼
Document Loaders
          │
          ▼
Text Chunking
          │
          ▼
Embeddings (all-MiniLM-L6-v2)
          │
          ▼
FAISS Vector Store
          │
          ▼
Retriever
          │
          ▼
Ollama (Llama 3.2)
          │
          ▼
Generated Answer
```

---

## Technologies Used

### AI / LLM

* Ollama
* Llama 3.2
* Sentence Transformers

### Frameworks

* LangChain
* Streamlit

### Vector Database

* FAISS

### Programming Language

* Python

### Document Processing

* PyPDFLoader
* Docx2txtLoader

---

## Project Structure

```text
financial-rag-assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│
├── vector_store/
│
└── src/
    ├── build_index.py
    ├── query.py
    └── rag_pipeline.py
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Nandakishore630/RAG_Application.git
cd RAG_Application
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download and install Ollama:

https://ollama.com

Pull Llama 3.2 model:

```bash
ollama pull llama3.2
```

Verify:

```bash
ollama list
```

---

## Build Vector Index

Place PDF and DOCX files inside the `data` folder.

Run:

```bash
python src/build_index.py
```

This will:

* Load documents
* Create chunks
* Generate embeddings
* Build FAISS index
* Save vector store

---

## Run Application

```bash
streamlit run app.py
```

Open the browser and ask questions about your financial documents.

---

## Example Questions

* What was Apple's revenue?
* What was Microsoft's revenue?
* Compare Apple's and Microsoft's revenue.
* What risks were highlighted in the report?
* What was the company's net income?

---

## Future Enhancements

* Source citations
* Chat history
* RAGAS evaluation metrics
* Re-ranking models
* Pinecone integration
* FastAPI backend
* Cloud deployment

---

## Author

**Y. Nanda Kishore Reddy**

B.Tech (CSE - Data Science)

Aspiring Data Scientist | Machine Learning & Generative AI Enthusiast
