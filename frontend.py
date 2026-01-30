import streamlit as st
import os
from extract import extract_all_text, clean_pdf_text
from chunking import chunking
from embedd import embed_sentences, embed_query
from db import create_faiss_index
from llm import generate_response
from search import search
import faiss
import redis
import warnings
warnings.filterwarnings('ignore')
# Connecting to Redis server on localhost
print('Connection Done!')
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

@st.cache_resource
def get_llmcache():
    return  SemanticCache(
                name="llmcache",                                          # underlying search index name
                redis_url="redis://localhost:6379",                       # redis connection url string
                distance_threshold=0.2,                                   # semantic cache distance threshold
                vectorizer=HFTextVectorizer("redis/langcache-embed-v1"),  # embdding model
            )
llmcache=get_llmcache()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
st.title("Document Q&A using RAG")

uploaded_file = st.file_uploader("Upload a file")

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File saved to {file_path}")

    if "faiss_index" not in st.session_state:
        raw_text = extract_all_text(file_path)
        cleaned_text = clean_pdf_text(raw_text)

        # Step 2: Chunk the cleaned text
        text_chunks = list(chunking(cleaned_text, chunk_size=500))

        # Step 3: Embed the text chunks
        chunk_embeddings = embed_sentences(text_chunks)

        # Step 4: Create FAISS index
        st.session_state.faiss_index = create_faiss_index(384,chunk_embeddings)
        st.session_state.text_chunks = text_chunks

faiss_index = st.session_state.get("faiss_index")
text_chunks = st.session_state.get("text_chunks")


prompt = st.chat_input("Ask a question about the uploaded document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()
    
if prompt:
    

    if response := llmcache.check(prompt=prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response[0]["response"] + " (cache hit)"})
    
    else:
    # st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
    # Step 5: Embed the user query
        query_embedding = embed_query(prompt)

        # Step 6: Search for relevant chunks
        top_k_indices = search(query_embedding, faiss_index, top_k=4)

        # Step 7: Retrieve top responses
        top_responses = "\n".join([text_chunks[idx] for idx in top_k_indices[0]])

        # Step 8: Generate response using LLM
        response=generate_response(prompt, top_responses, st.session_state.messages)

        st.session_state.messages.append({"role": "assistant", "content": response})

        llmcache.store(
            prompt=prompt,
            response=response,
        )
        
    # st.chat_message("assistant").write(response)

    # st.write(st.session_state.messages)

# --- Display all messages ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])


