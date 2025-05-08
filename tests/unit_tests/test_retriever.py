import unittest
from typing import Any
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_milvus import Milvus
from src.retriever import Retriever

DOC = Document(page_content="Mock content", metadata={"source": "mock_source"})


class MockUnvailableRunnable(Runnable):
    """Mock for general retriever runnable."""

    def __init__(self, message: str):
        self.exception_obj = Exception(message)

    def invoke(self, *args: Any, **kwargs: Any):
        raise self.exception_obj

    async def ainvoke(self, *args: Any, **kwargs: Any):
        raise self.exception_obj

    def _get_relevant_documents(self, *args: Any, **kwargs: Any):
        raise self.exception_obj


class MockMilvusRetriever(Milvus):
    """Mock for milvus client."""

    def __init__(self, retriever: Runnable = None):
        self.retriever = retriever

    def as_retriever(self, *args: Any, **kwargs: Any):
        return self.retriever


class MockReturnRunnable(Runnable):
    """Mock for runnable with mock document as return result."""

    def __init__(self):
        pass

    def invoke(self, *args: Any, **kwargs: Any):
        return [DOC]

    async def ainvoke(self, *args: Any, **kwargs: Any):
        return [DOC]

    def _get_relevant_documents(self, *args: Any, **kwargs: Any):
        return [DOC]


class TestRetrieverFallback(unittest.TestCase):

    def test_all_retriever_fallback(self):
        with patch(
            "src.retriever.TavilySearchAPIRetriever",
            return_value=MockUnvailableRunnable("Tavily is unavailable"),
        ), patch(
            "src.retriever.Milvus",
            return_value=MockMilvusRetriever(
                MockUnvailableRunnable("Milvus is unavailable")
            ),
        ):
            retriever = Retriever()
            question = "Testing question for all retrievers fallback"
            results = retriever().invoke(question)
            self.assertEqual(results, [])

    def test_tavily_fallback(self):
        """Test if Tavily retriever is unavailable."""
        with patch(
            "src.retriever.TavilySearchAPIRetriever",
            return_value=MockUnvailableRunnable("Tavily is unavailable"),
        ), patch(
            "src.retriever.Milvus",
            return_value=MockMilvusRetriever(MockReturnRunnable()),
        ):
            retriever = Retriever()
            question = "Testing question for Tavily fallback"
            results = retriever().invoke(question)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].page_content, DOC.page_content)
            self.assertEqual(results[0].metadata["source"], DOC.metadata["source"])

    def test_tavily_fallback(self):
        """Test if Milvus retriever is unavailable."""
        with patch(
            "src.retriever.TavilySearchAPIRetriever",
            return_value=MockReturnRunnable(),
        ), patch(
            "src.retriever.Milvus",
            return_value=MockMilvusRetriever(
                MockUnvailableRunnable("Milvus is unavailable")
            ),
        ):
            retriever = Retriever()
            question = "Testing question for Milvus fallback"
            results = retriever().invoke(question)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].page_content, DOC.page_content)
            self.assertEqual(results[0].metadata["source"], DOC.metadata["source"])
