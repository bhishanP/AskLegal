import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS

class PDFProcessor:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize PDF processor with embedding model
        
        :param embedding_model: Hugging Face embedding model name
        """
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        self.folder_path = "data/vectorstores/"  # Store vector DB in a separate folder
        os.makedirs(self.folder_path, exist_ok=True) 

    def load_pdf(self, pdf_path):
        """
        Load PDF and split into pages
        
        :param pdf_path: Path to PDF file
        :return: List of document pages
        """
        loader = PyMuPDFLoader(pdf_path) 
        return loader.load_and_split()

    def split_text(self, pages, chunk_size=1000, chunk_overlap=200):
        """
        Split document pages into smaller chunks
        
        :param pages: List of document pages
        :param chunk_size: Size of text chunks
        :param chunk_overlap: Overlap between chunks
        :return: List of split documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(pages)

    def create_vector_store(self, name, documents):
        """
        Create and save FAISS vector store
        
        :param name: Name of vector store
        :param documents: List of text documents
        :return: Path to saved vector store
        """
        try:
            vectorstore_faiss = FAISS.from_documents(documents, self.embedder)
            file_name = f"{name}"
            vectorstore_faiss.save_local(index_name=file_name, folder_path=self.folder_path)
            return os.path.join(self.folder_path, file_name)
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def process_pdf(self, pdf_path, vector_store_name='vector_space'):
        """
        Complete PDF processing pipeline
        
        :param pdf_path: Path to PDF file
        :param vector_store_name: Name for vector store
        :return: Path to created vector store
        """
        try:
            # Load PDF
            pages = self.load_pdf(pdf_path)
            
            # Split text
            split_docs = self.split_text(pages)
            
            # Create vector store
            return self.create_vector_store(vector_store_name, split_docs)
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None
