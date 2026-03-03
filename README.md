RAG ML Documentation Assistant

A Retrieval-Augmented Generation (RAG) application for querying official machine learning documentation. 

This project leverages LangChain, Groq API, and vector embeddings to provide precise answers from Python, PyTorch, and Scikit-learn official documentation.

The app allows users to input natural language questions and retrieves accurate responses from indexed documentation with contextual relevance.

Live App Link: https://7beshoyarnest-rag-ml-docum-ragml-documentation-assistant-mnjovc.streamlit.app/

Features

🔍 Contextual Q&A: Ask questions in natural language and get precise answers from official ML documentation.

🗂 Document Indexing: Automatically indexes multiple documentation sources using embeddings.

⚡ Fast Search: Uses FAISS vector store for similarity search and quick retrieval.

💬 Interactive UI: Built with Streamlit for an intuitive web interface.

🧠 Multi-Source Knowledge: Supports Python, PyTorch, and Scikit-learn documentation.

📄 Document Similarity View: Inspect retrieved document chunks for verification and transparency.

Supported Documentation

This assistant currently supports:

Python Built-in Functions

PyTorch nn.Module

PyTorch optim.Adam

Scikit-learn LogisticRegression

Scikit-learn RandomForestClassifier

Project Structure:

rag-ml-documentation-assistant/

│

├─ rag/

│   └─ ML_Documentation_Assistant.py     # Main Streamlit app

│
├─ requirements.txt                      # Python dependencies

├─ Live_App_url.txt                       # Live Streamlit app URL

├─ screenshots/                           # Example screenshots of the app

└─ README.md                              # This file

Installation:

Clone the repository:

git clone https://github.com/7BeshoyArnest/rag-ml-documentation-assistant.git

cd rag-ml-documentation-assistant

Create a virtual environment (optional but recommended):

python -m venv venv

venv\Scripts\activate      # Windows

Install dependencies:

pip install -r requirements.txt

Set your Groq and HuggingFace APIs keys:

GROQ_API_KEY=your_groq_api_key_here

HUGGINGFACE_API_KEY=your_hugginface_api_here

Usage

Navigate to the rag folder:

cd rag

Run the Streamlit app:

streamlit run ML_Documentation_Assistant.py

Enter your question in the input box. The app will:

Retrieve relevant document chunks.

Generate an answer using the Groq LLM.

Show the answer and the context used for retrieval.

Screenshots

Interactive Q&A interface

View retrieved documentation chunks

Technologies Used

Python 3.11+

Streamlit – Interactive web app framework

LangChain – LLM orchestration and retrieval chains

FAISS – Vector similarity search

HuggingFace Embeddings – Semantic document embeddings

Groq API – Large language model backend

Contributing

Contributions are welcome! To contribute:

Fork the repository.

Create a feature branch: git checkout -b feature-name.

Commit your changes: git commit -m "Add feature".

Push to the branch: git push origin feature-name.

Open a pull request.

License

This project is MIT licensed – see LICENSE for details.
