Create virtual enviornment : python -m venv myenv
activate the venv = myenv/Scripts/Activate

# Sanskrit Document Retrieval-Augmented Generation (RAG) System

## Objective
This project implements a CPU-based Retrieval-Augmented Generation (RAG) chatbot
that answers queries strictly based on Sanskrit documents.

## Features
- Sanskrit document ingestion and indexing
- Vector-based retrieval
- Local CPU-based LLM inference using Ollama
- Streamlit UI
- Sanskrit-only responses for greetings and document-based answers

## Tech Stack
- Python
- Streamlit
- FAISS
- SentenceTransformers
- Ollama (CPU-only)

## How to Run (Local)

1. Start Ollama:
   ```bash
   ollama serve

2. Run the streamlit app:

streamlit run app.py