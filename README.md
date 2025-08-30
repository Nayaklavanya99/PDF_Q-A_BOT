# PDF Q&A Assistant

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content using AI-powered question answering.

## Features

- **PDF Upload & Processing**: Upload PDF files and automatically extract text content
- **Intelligent Text Splitting**: Documents are split into manageable chunks for better processing
- **Semantic Search**: Uses HuggingFace embeddings to find relevant content
- **AI-Powered Answers**: Leverages OpenRouter API with GPT-4o-mini for accurate responses
- **Chat History**: Maintains conversation history for each PDF session
- **Real-time Processing**: Instant responses with loading indicators

## Prerequisites

- Python 3.8+
- OpenRouter API key

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install streamlit langchain-community langchain-text-splitters langchain-huggingface langchain-chroma python-dotenv requests
```

3. Create a `.env` file in the project directory:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Upload a PDF file using the file uploader

4. Wait for the PDF to be processed (text extraction and embedding creation)

5. Ask questions about the PDF content in the input field

6. View AI-generated answers based on the document content

## How It Works

1. **Document Loading**: PDFs are loaded using PyPDFLoader
2. **Text Splitting**: Content is split into 1000-character chunks with RecursiveCharacterTextSplitter
3. **Embeddings**: Text chunks are converted to embeddings using sentence-transformers/all-mpnet-base-v2
4. **Vector Storage**: Embeddings are stored in Chroma vector database
5. **Similarity Search**: User questions are matched against document chunks
6. **AI Response**: Relevant context is sent to OpenRouter API for answer generation

## Configuration

- **Embedding Model**: sentence-transformers/all-mpnet-base-v2
- **LLM Model**: openai/gpt-4o-mini (via OpenRouter)
- **Chunk Size**: 1000 characters
- **Search Results**: Top 3 most relevant chunks

## API Requirements

This application requires an OpenRouter API key. Get yours at [openrouter.ai](https://openrouter.ai/).

## File Structure

```
Pdf_Q_A/
├── app.py          # Main Streamlit application
├── .env            # Environment variables (create this)
└── README.md       # This documentation
```