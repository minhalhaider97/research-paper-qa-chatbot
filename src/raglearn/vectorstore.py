import numpy as np
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os
from src.raglearn.embedding import EmbeddingManger

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
        
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store
    
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        try:
            print(f"Initializing Vector store...")
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path = self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_document(self, all_document):
        ids = []
        embedding_list = []
        texts = []
        metadatas = []

        print("Embedding and chunking Started")

        emb_man = EmbeddingManger(self.model_name)
        embeddings = emb_man.generate_embedding(all_document)

        print("Adding Documents started...")
        for i,(doc, emb) in enumerate(zip(all_document, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_len'] = len(doc.page_content)
            metadatas.append(metadata)

            embedding_list.append(emb)

            texts.append(doc.page_content)

        try:
            self.collection.add(
                ids=ids,
                metadatas = metadatas,
                documents = texts,
                embeddings = embedding_list
            )
            print(f"Successfully added {len(all_document)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
