import streamlit as st
from rag import load_and_chunk_pdf, build_vectorstore, retrieve, generate_answer

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 Document RAG Chatbot")

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Sidebar for PDF Upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Check if it's a new file
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            # Reset chat history
            st.session_state.messages = []
            
            with st.spinner("Processing PDF and building vector store..."):
                file_bytes = uploaded_file.read()
                chunks = load_and_chunk_pdf(file_bytes)
                collection = build_vectorstore(chunks)
                st.session_state.collection = collection
                
                st.success(f"Successfully processed PDF! Created {len(chunks)} chunks.")

# Main Chat Interface

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 Sources"):
                for source in message["sources"]:
                    st.markdown(f"**Page {source['metadata']['page_number']}**")
                    st.markdown(f"_{source['text'][:300]}..._")
                    st.divider()

# React to user input
if prompt := st.chat_input("Ask a question about the uploaded document"):
    if st.session_state.collection is None:
        st.warning("Please upload a PDF document first before asking questions.")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve chunks
                retrieved_chunks = retrieve(prompt, st.session_state.collection)
                
                # Generate answer
                answer = generate_answer(prompt, retrieved_chunks)
                
                # Display answer
                st.markdown(answer)
                
                # Display sources
                with st.expander("📎 Sources"):
                    for source in retrieved_chunks:
                        st.markdown(f"**Page {source['metadata']['page_number']}**")
                        st.markdown(f"_{source['text'][:300]}..._")
                        st.divider()
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": retrieved_chunks
                })
