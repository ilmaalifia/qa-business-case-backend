import os

from app.utils import RRF_CONSTANT, get_bool_env, setup_logger
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import ArxivRetriever, PubMedRetriever
from langchain_community.retrievers.tavily_search_api import (
    SearchDepth,
    TavilySearchAPIRetriever,
)
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus

logger = setup_logger(__name__)
load_dotenv()


class Retriever:
    def __init__(
        self,
    ):
        retrievers = []
        self.top_k = int(os.getenv("TOP_K", "20"))
        if get_bool_env("MILVUS_ENABLE"):
            milvus_client = Milvus(
                embedding_function=[
                    HuggingFaceEmbeddings(model_name=os.getenv("DENSE_MODEL"))
                ],
                collection_name=os.getenv("MILVUS_COLLECTION", "pdf"),
                connection_args={
                    "uri": os.getenv("MILVUS_URI"),
                    "token": os.getenv("MILVUS_TOKEN"),
                },
                vector_field=["dense_vector", "sparse_vector"],
                search_params=[
                    {
                        "metric_type": "COSINE",
                    },
                    {
                        "metric_type": "BM25",
                    },
                ],
                builtin_function=BM25BuiltInFunction(
                    input_field_names="text",
                    output_field_names="sparse_vector",
                    analyzer_params={"type": "english"},
                    function_name="bm25_function",
                ),
            )

            self.milvus = milvus_client.as_retriever(
                search_kwargs={
                    "k": self.top_k,
                    "ranker_type": "rrf",
                    "ranker_params": {"k": RRF_CONSTANT},
                    "group_by_field": "source",
                    "group_size": 5,
                },
                tags=["milvus"],
            ).with_fallbacks(self.__retriever_fallback())
            retrievers.append(self.milvus)

        if get_bool_env("TAVILY_ENABLE"):
            self.tavily = TavilySearchAPIRetriever(
                k=self.top_k,
                search_depth=SearchDepth.ADVANCED,
                tags=["tavily"],
            ).with_fallbacks(self.__retriever_fallback())
            retrievers.append(self.tavily)

        if get_bool_env("ARXIV_ENABLE"):
            self.arxiv = ArxivRetriever(
                load_max_docs=self.top_k,
                get_full_documents=False,
                tags=["arxiv"],
            ).with_fallbacks(self.__retriever_fallback())
            retrievers.append(self.arxiv)

        if get_bool_env("PUBMED_ENABLE"):
            self.pubmed = PubMedRetriever(
                api_key=os.getenv("PUBMED_API_KEY"),
                top_k_results=self.top_k,
                sleep_time=0.5,
                tags=["pubmed"],
            ).with_fallbacks(self.__retriever_fallback())
            retrievers.append(self.pubmed)

        self.retriever = self.__setup_ensemble_retriever(retrievers)

    def __setup_ensemble_retriever(self, retrievers: list):
        return EnsembleRetriever(
            retrievers=retrievers,
            c=RRF_CONSTANT,
            id_key=(
                None
                if (get_bool_env("ARXIV_ENABLE") or get_bool_env("PUBMED_ENABLE"))
                else "source"
            ),
            tags=["ensemble"],
        )

    def __call__(self):
        return self.retriever

    def __retriever_fallback(self):
        return [RunnableLambda(lambda x: [])]
