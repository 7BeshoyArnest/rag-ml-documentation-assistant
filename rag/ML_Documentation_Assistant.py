import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

## Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

# Build Vector Store
if "vector" not in st.session_state:

    st.info("Building documentation index...")

    # Use Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    st.session_state.loader = WebBaseLoader(
        "https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/core/langchain_core/prompts/chat.py"
    )
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_documents = (
        st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    )

    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, embeddings
    )

    st.success("Vector store built successfully.")

# LLM Setup
st.title("ChatGroq Demo")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<contxt>
{context}
<context>
Question:{input}
"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User Input
user_prompt = st.text_input("Input your prompt here")

if user_prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    elapsed = time.process_time() - start

    st.write(response['answer'])
    st.caption(f"⏱ Response time: {elapsed:.2f} seconds")

    # Show relevant document chunks
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------")
