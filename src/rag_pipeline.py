import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ✅ Paths
BASE_PATH = "C:/Users/HP/Knowledge-Assistant/data"
VECTORSTORE_PATH = os.path.join(BASE_PATH, "vectorstore")

# ✅ Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load Chroma vector DB
db = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)

# ✅ Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# ✅ LLM (you can switch to "gpt-4o-mini" or any supported model)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ✅ Build QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

print("🚀 Knowledge Assistant is ready! Ask a question (type 'exit' to quit)\n")

# ✅ Interactive Q&A loop
while True:
    query = input("❓ Ask: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Goodbye!")
        break
    try:
        result = qa.run(query)
        print(f"💡 Answer: {result}\n")
    except Exception as e:
        print(f"⚠️ Error: {e}\n")
