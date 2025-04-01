import streamlit as st
import fitz  # PyMuPDF for PDF handling
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# Load Gemini API Key
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
else:
    raise ValueError("API key for Gemini AI is missing. Please set GENAI_API_KEY in your .env file.")

# Define Constants
MAX_CONTEXT_LENGTH = 2000
SAVE_FOLDER = "saved_pdfs"
os.makedirs(SAVE_FOLDER, exist_ok=True)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # Matches embedding model output size
faiss_index = faiss.IndexFlatL2(dimension)
doc_chunks, doc_sources = [], []

# Utility Functions
def extract_text_from_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text("text") for page in doc])
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def add_to_faiss(chunk, embedding, source):
    global faiss_index, doc_chunks, doc_sources
    doc_chunks.append(chunk)
    doc_sources.append(source)
    faiss_index.add(np.array([embedding], dtype="float32"))

def search_faiss(query_embedding, top_k=5):
    if len(doc_chunks) == 0:
        return [], None

    distances, indices = faiss_index.search(np.array([query_embedding], dtype="float32"), top_k)
    doc_results = defaultdict(list)

    for i in indices[0]:
        if i < len(doc_chunks):
            doc_results[doc_sources[i]].append(doc_chunks[i])

    if doc_results:
        most_relevant_doc = max(doc_results, key=lambda k: len(doc_results[k]))
        return doc_results[most_relevant_doc], most_relevant_doc

    return [], None

# Helper function to truncate context
def truncate_context(context, max_length=MAX_CONTEXT_LENGTH):
    return context[:max_length] + "..." if len(context) > max_length else context

# Streamlit UI
st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="wide")
st.markdown("<h1 style='text-align: center;'>📄 AskDoc</h1>", unsafe_allow_html=True)

# Sidebar for Chat History and Uploaded Documents
st.sidebar.header("📜 Recent Chat History")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store only user questions in the sidebar
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        if st.sidebar.button(message["content"], key=f"sidebar_{i}"):
            st.session_state.selected_message_index = i

# Uploaded Documents Section
st.sidebar.subheader("📄 Uploaded Documents")
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = set()

for doc in st.session_state.uploaded_docs:
    st.sidebar.markdown(f"- {doc}")

# File Upload Section
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        save_path = os.path.join(SAVE_FOLDER, uploaded_file.name)

        # Save the uploaded file locally
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the PDF (extract text and embeddings)
        chunks = extract_text_from_pdf(save_path)
        chunk_embeddings = [embed_model.encode(chunk).tolist() for chunk in chunks if chunk.strip()]

        for chunk, embedding in zip(chunks, chunk_embeddings):
            add_to_faiss(chunk, embedding, uploaded_file.name)

        st.session_state.uploaded_docs.add(uploaded_file.name)  # Store the uploaded file names
        progress_bar.progress((idx + 1) / total_files)

    st.success(f"{len(uploaded_files)} PDFs uploaded, saved locally, and processed successfully!")
    st.experimental_rerun()  # Refresh the page to update the sidebar

# Chat Interface
st.markdown("## 💬 Ask a Question")

# Display Previous Messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "selected_message_index" in st.session_state and st.session_state.selected_message_index == i:
            st.markdown("", unsafe_allow_html=True)

# Handle User Input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
    query_embedding = embed_model.encode(prompt).tolist()
    results, source_doc = search_faiss(query_embedding, top_k=5)

    model = genai.GenerativeModel("gemini-1.5-flash")

    if results:
        relevant_text = "\n.join(results)
        truncated_relevant_text = truncate_context(relevant_text)

        ai_prompt = f"""
        You are an AI assistant. Answer the user's question based on the given PDF context along with your general explanation.
        **Context from {source_doc}:**  
        {truncated_relevant_text}

        **User Question:**  
        {prompt}
        """

        response = model.generate_content(ai_prompt)
        answer = response.text
    else:
        response = model.generate_content(f"No relevant content found in the uploaded documents. Please upload more documents for better context")
        answer = response.text

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Refresh chat history immediately
    st.rerun()