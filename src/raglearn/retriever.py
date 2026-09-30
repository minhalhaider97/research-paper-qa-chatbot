from src.raglearn.embedding import EmbeddingManger
from src.raglearn.vectorstore import VectorStore
import numpy as np
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class RagRetriever:
    def __init__(self, vectorstore: VectorStore, embedding_manager: EmbeddingManger ):
        self.vectorstore = vectorstore
        self.embedding_man = embedding_manager

    def retrieve(self, query: str , top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        ## converting query into embedding

        query_emb = self.embedding_man.generate_query_embedding(query)

        ## Result

        results = self.vectorstore.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )

        retrieved_docs=[]
        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]

            for i,(doc_id, document, metadata, distance) in enumerate (zip(ids, documents, metadatas, distances)):
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id" : doc_id,
                        "content" : document,
                        "metadata" : metadata,
                        "rank" : i+1,
                        "similarity_score" : similarity_score,
                        "distance" : distance}
                    )

            print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
        else:
            print("No documents found")

        return retrieved_docs