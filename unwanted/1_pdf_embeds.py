from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

curr_dir  = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curr_dir,"docs","ml.pdf")
persistent_dir = os.path.join(curr_dir,"db")



# Check if the Chroma vector store already exists
if not os.path.exists(persistent_dir):
    print("Persistent directory not found. Creating vector store...")
    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    ## Read the content from the pdf
    loader = PyPDFLoader(file_path,mode="page")
    docs = []
    docs_lazy = loader.load()
    for doc in docs_lazy:
        docs.append(doc)
    print(f"number of pages in the pdf: {len(docs)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    print(f"number of chunks after splitting: {len(split_docs)}")


    #create embedding from huggingface
    print("\n--- Creating embeddings ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("\n--- Finished creating embeddings ---")

    # Create and persist the Chroma vector store
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(split_docs,embeddings,persist_directory=persistent_dir)
    print("\n--- Finished creating vector store ---")
else:
    print("Vector store already exists. No need to initialize.")