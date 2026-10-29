import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

#  (GROQ_API_KEY)
load_dotenv()

# Initialize models and clients
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()  # local in-memory
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_and_chunk_pdf(file_bytes: bytes, chunk_size: int = 500, overlap: int = 50):
    """
    Loads a PDF from bytes, extracts text, and chunks it using word-based chunking.
    Tracks page numbers for each chunk.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        words = text.split()
        
        # Word-based chunking with overlap
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "metadata": {"page_number": page_num + 1} # 1-indexed page numbers
            })
            i += chunk_size - overlap
            
    return chunks

def build_vectorstore(chunks, collection_name="rag_collection"):
    """
    Embeds chunks and stores them in ChromaDB.
    Deletes the collection if it already exists to avoid conflicts on new uploads.
    """
    import chromadb.errors
    try:
        chroma_client.delete_collection(name=collection_name)
    except chromadb.errors.NotFoundError:
        pass # Collection does not exist yet
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # Use cosine similarity
    )
    
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Generate embeddings
    embeddings = embedding_model.encode(texts).tolist()
    
    # Store in ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection

def retrieve(query: str, collection, top_k: int = 3):
    """
    Retrieves the top_k most similar chunks for the given query.
    """
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    retrieved_chunks = []
    if results['documents'] and len(results['documents']) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        for doc, meta in zip(docs, metas):
            retrieved_chunks.append({
                "text": doc,
                "metadata": meta
            })
            
    return retrieved_chunks

def generate_answer(query: str, retrieved_chunks: list):
    """
    Generates an answer using Groq API based on the retrieved context.
    """
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_text += f"--- Document Chunk {i+1} (Page {chunk['metadata']['page_number']}) ---\n"
        context_text += chunk['text'] + "\n\n"

    system_prompt = (
        "You are a helpful assistant. You must answer the user's question based ONLY on the provided context documents. "
        "When you provide an answer, you must mention the page number(s) from which you extracted the information. "
        "If the answer is not found in the context documents, you must exactly say: 'I could not find this in the document'."
    )
    
    user_prompt = f"Context Documents:\n{context_text}\nQuestion: {query}"
    
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="openai/gpt-oss-120b",
        temperature=0.2
    )
    
    return response.choices[0].message.content
