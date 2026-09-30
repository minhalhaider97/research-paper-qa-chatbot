from src.raglearn.retriever import RagRetriever
from src.raglearn.embedding import EmbeddingManger
from src.raglearn.vectorstore import VectorStore
from langchain_groq import ChatGroq
import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

class Search:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.history = []
        self.llm = ChatGroq(model=model_name, temperature=0.1,max_tokens=1024)
        self.emb_manager = EmbeddingManger()
        self.vector_store = VectorStore()

    def ask_query(self, query: str , top_k:int = 5) -> Dict[str, Any]:

        ragret = RagRetriever(vectorstore=self.vector_store, embedding_manager=self.emb_manager)
        results = ragret.retrieve(query=query)

        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
            } for doc in results]

            # Streaming answer simulation
            prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""

            response = self.llm.invoke([prompt.format(context=context, question=query)])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        summary = None
        if answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'question': query,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': query,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
        }