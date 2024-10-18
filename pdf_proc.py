import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

# Load a local embedding model from Hugging Face
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(name, documents):
    vectorstore_faiss = FAISS.from_documents(documents, embedder)
    file_name = f"{name}"
    folder_path = "./"  # Save locally in the current directory
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    return os.path.join(folder_path, file_name)

def main():
    st.title("Document Search Engine")
    
    loader = PyPDFLoader('data/Constitution-of-Nepal.pdf')
    pages = loader.load_and_split()

    splitted_docs = split_text(pages, 1000, 200)

    # Create vector store using local embeddings
    result = create_vector_store('vector_space', splitted_docs)

    st.write("Vector Store Created Successfully:", result)

if __name__ == "__main__":
    main()
