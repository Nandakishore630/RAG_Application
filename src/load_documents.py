from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\nanda\Desktop\financial_rag-assistant\data\apple-25.pdf")

documents = loader.load()

print(f"Pages: {len(documents)}")

print(documents[0].page_content[:500])