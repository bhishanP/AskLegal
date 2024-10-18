import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub  
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


folder_path = "./"

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm():
    llm = HuggingFaceHub(repo_id="EleutherAI/gpt-neo-2.7B", 
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_kwargs={"max_length": 512, 'max_new_tokens': 100}) 
    return llm

# get_response() using local LLM and vectorstore
def get_response(llm, vectorstore, question):
    ## Create prompt / template
    prompt_template = """
    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    answer = qa({"query": question})
    return answer['result']

## main method
def main():
    st.header("Chat with PDF Demo using Local LLM and RAG")


    ## Load FAISS index locally
    faiss_index = FAISS.load_local(
        index_name="vector_space",
        folder_path=folder_path,
        embeddings=embedder,
        allow_dangerous_deserialization=True
    )

    question = st.text_input("Please ask your question")
    
    if st.button("Ask Question"):
        with st.spinner("Querying..."):
            llm = get_llm()
            st.write(get_response(llm, faiss_index, question))
            st.success("Done")

if __name__ == "__main__":
    main()
