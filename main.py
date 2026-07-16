from fastapi import FastAPI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import pipeline

# -----------------------------
# LOAD PDF
# -----------------------------
loader = PyPDFLoader("sample.pdf")
documents = loader.load()
print("✅ PDF Loaded")

# -----------------------------
# SPLIT
# -----------------------------
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
print(f"✅ Chunks: {len(chunks)}")

# -----------------------------
# EMBEDDINGS
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(chunks, embeddings)
print("✅ Vector DB Ready")

# -----------------------------
# SMALL TRANSFORMER (SAFE)
# -----------------------------
print("🔥 Loading small transformer...")
llm = pipeline(
    "text-generation",
    model="distilgpt2"
)

# -----------------------------
# FASTAPI
# -----------------------------
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API working 🚀"}

@app.get("/ask")
def ask(question: str):
    results = db.similarity_search(question, k=1)
    context = results[0].page_content

    prompt = f"""
Answer the question based on the context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm(
        prompt,
        max_new_tokens=50,
        do_sample=False
    )

    return {"answer": response[0]["generated_text"]}