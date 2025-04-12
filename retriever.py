from typing import List

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.tavily_search_api import (
    SearchDepth,
    TavilySearchAPIRetriever,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

load_dotenv()

URI = "./milvus_demo.db"


class Retriever:
    def __init__(
        self,
        top_k_milvus: int = 20,
        top_k_tavily: int = 20,
    ):
        self.model_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        self.milvus = Milvus(
            embedding_function=self.model_embeddings,
            collection_name="pdf",
            connection_args={"uri": URI},
        )
        self.tavily = TavilySearchAPIRetriever(
            k=top_k_tavily, search_depth=SearchDepth.ADVANCED
        )
        self.retriever = EnsembleRetriever(
            retrievers=[
                self.milvus.as_retriever(search_kwargs={"k": top_k_milvus}),
                self.tavily,
            ],
            weights=[0.5, 0.5],
        )

    def retrieve(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def __call__(self):
        return self.retriever | self.format_docs


if __name__ == "__main__":
    r = Retriever()
    query = "What is virtual power plant?"
    result = r.retrieve(query)
    print(f"Query: {query}")
    print(f"Result: {result}")
