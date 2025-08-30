import os
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")
url = "https://openrouter.ai/api/v1/chat/completions"

curr_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(curr_dir, "db")

# Load existing vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

def query_pdf(question):
    # Retrieve relevant documents
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context. If you don't know the answer from the context, just say that you don't know."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "PDF Q&A"
    }
    
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": messages
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"

# Example usage
if __name__ == "__main__":
    question = "Types of ML?"
    answer = query_pdf(question)
    print(answer)