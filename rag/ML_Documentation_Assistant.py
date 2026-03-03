import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

# Official Documentation URLs
urls = [
    # Python
    "https://docs.python.org/3/library/functions.html",

    # PyTorch
    "https://pytorch.org/docs/stable/generated/torch.nn.Module.html",
    "https://pytorch.org/docs/stable/generated/torch.optim.Adam.html",

     # Scikit-learn
    "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
]

# Build Vector Store
if "vector" not in st.session_state:
    st.info("Building documentation index...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    all_docs = []

    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_documents = text_splitter.split_documents(all_docs)

    st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
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
