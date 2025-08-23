import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Free embeddings

# Paths
DATA_PATH = "C:/Users/HP/Knowledge-Assistant/data"
VECTORSTORE_PATH = "C:/Users/HP/Knowledge-Assistant/data/vectorstore"

def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"📄 Loading {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())
    print(f"✅ Loaded {len(docs)} documents from PDFs")
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(splits)} chunks")
    return splits

def save_to_chroma(splits):
    print("💾 Saving to Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅ Free model
    db = Chroma.from_documents(splits, embeddings, persist_directory=VECTORSTORE_PATH)
    db.persist()
    print("✅ Saved embeddings to Chroma")

if __name__ == "__main__":
    docs = load_documents()
    splits = split_documents(docs)
    save_to_chroma(splits)



