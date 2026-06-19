from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = PyPDFLoader(r"C:\Users\nanda\Desktop\financial_rag-assistant\data\apple-25.pdf")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Total Chunks:", len(chunks))

print(chunks[0].page_content)