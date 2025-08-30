import streamlit as st
import os
import tempfile
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📚")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    
    embeddings = get_embeddings()
    db = Chroma.from_documents(split_docs, embeddings)
    
    os.unlink(tmp_file_path)
    return db

def query_openrouter(question, context):
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not API_KEY:
        return "OpenRouter API key not found in environment variables."
    
    messages = [
        {"role": "system", "content": "Answer based on the provided context. If you don't know the answer from the context, say you don't know."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "PDF Q&A"
    }
    
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": messages
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("📚 PDF Q&A Assistant")
st.markdown("Upload a PDF and ask questions about its content!")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    pdf_name = uploaded_file.name
    st.success(f"📄 Loaded: {pdf_name}")
    
    with st.spinner("Processing PDF..."):
        if 'db' not in st.session_state or st.session_state.get('current_pdf') != pdf_name:
            st.session_state.db = process_pdf(uploaded_file)
            st.session_state.current_pdf = pdf_name
            st.session_state.chat_history = []  # Clear chat history for new PDF
    
    st.success("✅ PDF processed successfully!")
    
    # Display chat history (readonly)
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        chat_container = st.container()
        with chat_container:
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")
                st.markdown("---")
    
    # Input area with send button
    st.markdown("### Ask a Question")
    
    with st.form(key="question_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input(
                "Your question:",
                key="question_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_clicked = st.form_submit_button("Send", type="primary")
    
    # Handle question submission
    if send_clicked and question.strip():
        with st.spinner("Searching for answer..."):
            docs = st.session_state.db.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            answer = query_openrouter(question, context)
            
            # Add to chat history
            st.session_state.chat_history.append((question, answer))
            
            # Rerun to update the display
            st.rerun()

else:
    st.info("👆 Please upload a PDF file to get started!")