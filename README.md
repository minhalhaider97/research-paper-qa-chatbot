# ResearchPaper Q/A Chatbot

An intelligent Retrieval-Augmented Generation (RAG) pipeline and chatbot application designed to query, search, and extract precise answers from research papers and technical documents.

---

## Features
* **Document Processing & Chunking:** Automatically splits and chunks long research papers using recursive character text splitting.
* **Vector Embeddings:** Generates dense vector embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`) for semantic search capabilities[cite: 1].
* **Vector Store Integration:** Efficient document storage and similarity retrieval.
* **LLM-Powered Q/A:** Integrates with Groq and LangChain to provide accurate, context-aware answers based on retrieved research paper segments.

---

## Project Structure
```text
RAGlearn/
│
├── data/                  # Storage for research papers and documents
├── src/                   # Source code modules
│   ├── raglearn/
│   │   ├── embedding.py   # Embedding management and chunking logic[cite: 1]
│   │   ├── dataloader.py  # Data loading utilities
│   │   ├── retriever.py   # Document retrieval logic
│   │   ├── search.py      # Search functionality
│   │   └── vectorstore.py # Vector database management
│   │
├── app.py                 # Main application entry point
├── basicRag.ipynb         # Experimental RAG notebook
├── ragpipeline.ipynb      # End-to-end RAG pipeline notebook
├── pyproject.toml         # Project dependencies and metadata
├── requirement.txt        # Python package requirements
└── .env                   # Environment variables (API keys - ignored by git)
