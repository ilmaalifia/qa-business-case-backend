import os
from typing import List

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.tavily_search_api import (
    SearchDepth,
    TavilySearchAPIRetriever,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from utils import setup_logger

logger = setup_logger(__name__)
load_dotenv()


class Retriever:
    def __init__(
        self,
        top_k_milvus: int = 20,
        top_k_tavily: int = 20,
    ):
        self.model_embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("DENSE_MODEL")
        )

        self.milvus = Milvus(
            embedding_function=[self.model_embeddings],
            collection_name=os.getenv("MILVUS_COLLECTION", "pdf"),
            connection_args={
                "uri": os.getenv("MILVUS_URI"),
                "token": os.getenv("MILVUS_TOKEN"),
            },
            vector_field=["dense_vector", "sparse_vector"],
            builtin_function=BM25BuiltInFunction(
                input_field_names="text",
                output_field_names="sparse_vector",
                analyzer_params={"type": "english"},
                function_name="bm25_function",
            ),
            search_params=[
                {
                    "metric_type": "COSINE",
                },
                {
                    "metric_type": "BM25",
                },
            ],
        )

        self.tavily = TavilySearchAPIRetriever(
            k=top_k_tavily,
            search_depth=SearchDepth.ADVANCED,
        )

        self.retriever = EnsembleRetriever(
            retrievers=[
                self.milvus.as_retriever(
                    search_kwargs={
                        "k": top_k_milvus,
                        "ranker_type": "rrf",
                        "ranker_params": {"k": 60},
                    }
                ),
                self.tavily,
            ],
        )

    def retrieve(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)

    # TODO: Manage doc positioning
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def __call__(self):
        return self.retriever | self.format_docs


if __name__ == "__main__":
    r = Retriever()
    query = "What is virtual power plant?"
    result = r.retrieve(query)
    logger.info(f"Query: {query}")
    logger.info(f"Result: {result}")
