import streamlit as st
import os
from extract import extract_all_text, clean_pdf_text
from chunking import chunking
from embedd import embed_sentences, embed_query
from db import create_faiss_index
from llm import generate_response
from search import search
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Upload a file")

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File saved to {file_path}")

text_input = st.text_input("Enter your query:")

prompt = st.chat_input("Ask a question about the uploaded document.")

if st.button("Get Response") and text_input:
   
    # Step 1: Extract and clean text from PDF
    raw_text = extract_all_text(uploaded_file.name)
    cleaned_text = clean_pdf_text(raw_text)

    # Step 2: Chunk the cleaned text
    text_chunks = list(chunking(cleaned_text, chunk_size=500))

    # Step 3: Embed the text chunks
    chunk_embeddings = embed_sentences(text_chunks)

    # Step 4: Create FAISS index
    faiss_index = create_faiss_index(384,chunk_embeddings)

    # Step 5: Embed the user query
    query_embedding = embed_query(text_input)

    # Step 6: Search for relevant chunks
    top_k_indices = search(query_embedding, faiss_index, top_k=4)

    # Step 7: Retrieve top responses
    top_responses = "\n".join([text_chunks[idx] for idx in top_k_indices[0]])

    # Step 8: Generate response using LLM
    response=generate_response(text_input, top_responses)
    st.write("Response:")
    st.write(response)


