# 📚 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about it. Built from scratch using ChromaDB, Sentence Transformers, and Groq LLM.

## 🚀 Overview
An interactive Document AI assistant that allows users to upload any PDF document and ask contextual questions. Answers are retrieved via vector search and generated strictly from document excerpts with page-level citations.

## ✨ Features
- **PDF Ingestion**: Upload text-based PDF documents
- **Context-Grounded QA**: Responses strictly grounded in the document content
- **Source Citations**: Displays page numbers and chunk previews for every answer
- **Multi-turn Chat**: Retains context across questions within the session

## 🛠️ Tech Stack
- **Embeddings** — Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store** — ChromaDB (in-memory, cosine similarity)
- **LLM** — Groq API (`llama-3.3-70b-versatile`)
- **UI** — Streamlit
- **PDF Parsing** — PyMuPDF (`fitz`)

## ⚙️ How It Works
1. PDF is loaded and split into overlapping word-based chunks.
2. Each chunk is embedded using Sentence Transformers.
3. Chunks and embeddings are stored in ChromaDB with page metadata.
4. On query, the question is embedded and the top 3 similar chunks are retrieved.
5. Retrieved chunks are passed as context to the Groq LLM.
6. The LLM synthesizes an answer strictly from the retrieved context, citing exact page numbers.

## 🔧 Run Locally

```bash
git clone https://github.com/Yuvrajj13/document-rag-chatbot.git
cd document-rag-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
GROQ_API_KEY=your_groq_api_key

Run:
```bash
streamlit run app.py
```

## 📁 Project Structure
rag-chatbot/
├── app.py              # Streamlit UI
├── rag.py              # Core RAG logic
├── requirements.txt
└── .env                # Not committed

## 🔑 Environment Variables
| Variable | Description |
|---|---|
| GROQ_API_KEY | Your Groq API key (get it free at console.groq.com) |
