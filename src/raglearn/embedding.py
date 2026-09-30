from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

class EmbeddingManger:
    def __init__(self , model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._loadmodel()

    def chunk(self,all_documents,chunk_size=1000,chunk_overlap=200):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        split_docs = splitter.split_documents(all_documents)
        print(f"Split {len(self.all_documents)} documents into {len(split_docs)} chunks")

        return split_docs

    def _loadmodel(self):
        print(f"Loading Model {self.model_name}....")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model Loaded Successfully and model dimensions are {self.model.get_embedding_dimension()}")

    def generate_embedding(self, all_documents)->np.ndarray:
        split_docs = self.chunk(all_documents)
        if not self.model:
            raise ValueError("Embedding Model not found")

        print("Embedding Started....")
        texts= [doc.page_content for doc in split_docs]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"generated embeddings with shape{embeddings.shape}")


        return embeddings

    def generate_query_embedding(self, query: str):
        if not self.model:
            raise ValueError("Embedding Model not found")
        
        query_emb = self.model.encode(query)

        return query_emb








    
        