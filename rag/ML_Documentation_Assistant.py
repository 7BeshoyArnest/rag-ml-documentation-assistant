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
from bs4 import BeautifulSoup

# Environment Setup
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

st.set_page_config(page_title="ML Documentation Assistant", layout="wide")
st.title("🚀 ML Documentation Assistant")

st.markdown("""
Ask questions about:
- Python built-in functions
- PyTorch modules
- Scikit-learn models
""")

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
if "vectors" not in st.session_state:
    st.info("Building documentation index... This runs only once.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    all_docs = []

    # Load all docs
    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()

        # Clean HTML content
        for doc in docs:
            soup = BeautifulSoup(doc.page_content, "html.parser")
            doc.page_content = soup.get_text()

        all_docs.extend(docs)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_documents = text_splitter.split_documents(all_docs)

    st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
    st.session_state.urls = urls
    st.success("Vector store built successfully.")

# LLM Setup
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

prompt_template = ChatPromptTemplate.from_template(
"""
You are an expert Machine Learning Documentation Assistant.

Answer ONLY using the provided documentation context.
If the answer is not found in the context, say:
"I cannot find this in the official documentation."

<context>
{context}
</context>

Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User Input
user_prompt = st.text_input("💬 Ask your question:")

if user_prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    elapsed = time.process_time() - start

    st.subheader("📌 Answer")
    st.write(response["answer"])
    st.caption(f"⏱ Response time: {elapsed:.2f} seconds")

    # Show relevant document chunks
    with st.expander("🔎 Retrieved Documentation Chunks"):
        for i, doc in enumerate(response["context"]):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc.page_content[:1000])
            st.write("---")

    # Show sources
    with st.expander("🌐 Documentation Sources Used"):
        for url in st.session_state.urls:
            st.write(url)
