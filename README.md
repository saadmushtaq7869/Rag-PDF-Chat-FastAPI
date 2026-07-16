# 📄 AI Document Chat (RAG System)

This project is a simple AI-powered document question-answering system using:

- LangChain
- HuggingFace Transformers
- Chroma Vector Database
- FastAPI

## 🚀 Features

- Load PDF documents
- Split into chunks
- Generate embeddings
- Store in vector database
- Ask questions via API

## ⚠️ Note

Due to local system limitations (Mac M1 + PyTorch), transformer models may cause crashes.

For stable usage:
- Use smaller models (distilgpt2)
- Or integrate OpenAI API

## ▶️ Run

```bash
uvicorn main:app --workers 1
