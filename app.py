import streamlit as st
import os

from backend.pdf_processor import PDFProcessor
from backend.chatbot import PDFChatbot

def main():
    """Streamlit Chatbot Application"""
    st.title("PDF Chatbot")

    # Initialize chatbot once
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()

    # Clear chat history
    if st.button('Clear Chat History'):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    # Load vector store once
    if 'vector_store' not in st.session_state:
        try:
            st.session_state.vector_store = st.session_state.chatbot.load_vector_store()
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return

    # Chat input
    user_question = st.text_input("Ask a question about the document:")

    # Process chat
    if user_question:
        try:
            with st.spinner("Processing..."):
                response = st.session_state.chatbot.get_response(
                    user_question, st.session_state.vector_store
                )

            # Display answer
            if 'answer' in response:
                st.write("Assistant:", response['answer'])
            else:
                st.warning("No response generated.")

            # Display source documents
            if 'source_documents' in response:
                with st.expander("Source Documents"):
                    for doc in response['source_documents']:
                        st.write(doc.page_content)
        except Exception as e:
            st.error(f"Error processing question: {e}")

def create_vector_store():
    """Utility function to create vector store"""
    st.title("Create Vector Store")

    # Ensure directory exists
    os.makedirs("data", exist_ok=True)

    # PDF file uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        pdf_path = f"data/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process PDF
        processor = PDFProcessor()
        with st.spinner("Processing PDF..."):
            result = processor.process_pdf(pdf_path)
            st.success(f"Vector store created: {result}")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Chatbot", "Create Vector Store"])

if app_mode == "Chatbot":
    main()
else:
    create_vector_store()
