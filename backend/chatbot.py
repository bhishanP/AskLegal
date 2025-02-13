import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

class PDFChatbot:
    def __init__(
        self, 
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="llama3-8b-8192"
    ):
        """
        Initialize chatbot with embedding and language models
        
        :param embedding_model: Hugging Face embedding model
        :param llm_model: Groq language model
        """
        load_dotenv()
        
        # Initialize embeddings
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize language model
        self.llm = ChatGroq(
            temperature=0.2,
            model_name=llm_model,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Vector store path
        self.folder_path = "./"
        
        # Conversation chain
        self.conversation = None

    def load_vector_store(self, index_name="vector_space"):
        """
        Load existing FAISS vector store
        
        :param index_name: Name of vector store index
        :return: Loaded FAISS vector store
        """
        return FAISS.load_local(
            index_name=index_name,
            folder_path=self.folder_path,
            embeddings=self.embedder,
            allow_dangerous_deserialization=True
        )

    def create_conversation_chain(self, vectorstore):
        """
        Create conversational retrieval chain
        
        :param vectorstore: FAISS vector store
        :return: Conversational retrieval chain
        """
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key= "answer",
        )
        
        self.conversation = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.get_custom_prompt()}
        )
        
        return self.conversation
    def get_custom_prompt(self):
        """
        Creates a custom prompt template to ensure retrieval context is used.
        """
        from langchain.prompts import PromptTemplate

        return PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the following retrieved documents to answer the question:\n\n"
                "{context}\n\n"
                "Question: {question}\n\n"
                "If the documents do not provide enough information, say 'I don't know'."
            )
        )
    def get_response(self, question, vectorstore=None):
        """
        Get response for a given question
        
        :param question: User's question
        :param vectorstore: Optional vector store (will load default if not provided)
        :return: Dictionary with answer and source documents
        """
        # Load vector store if not provided
        if vectorstore is None:
            vectorstore = self.load_vector_store()
        
        # Create conversation chain if not exists
        if self.conversation is None:
            self.create_conversation_chain(vectorstore)
        
        # Get response
        return self.conversation({"question": question})