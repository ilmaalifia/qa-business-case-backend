import unittest
from unittest.mock import patch

import httpx
from app.generator import NO_ANSWER_PROMPT, Generator
from app.utils import convert_document_to_additional_source
from langchain_core.documents import Document
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai import RateLimitError

DOCS = [
    Document(
        page_content="This is a mock document for testing purposes.",
        metadata={"source": "https://mock_source.com/mock.pdf", "page": 1},
    ),
    Document(
        page_content="This is another mock document for testing purposes.",
        metadata={"source": "https://mock_source_2.com/mock.pdf", "page": 2},
    ),
]

CONTEXT = """
Source: https://mock_source.com/mock.pdf
Page: 1
Information: This is a mock document for testing purposes.

---

Source: https://mock_source_2.com/mock.pdf
Page: 2
Information: This is another mock document for testing purposes.
"""

LLM_FALLBACK = {
    "answer": "Unable to answer the question due to error. Please try again.",
    "citations": [],
    "additional_sources": [],
}


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = Generator()

    def test_format_docs_as_context(self):
        context = self.generator.format_docs_as_context(DOCS)
        self.assertEqual(context, CONTEXT)

    def test_prompt(self):
        """Test if the prompt contains system and huma message role."""
        prompt = self.generator.get_prompt()
        self.assertIsInstance(prompt.messages[0], SystemMessagePromptTemplate)
        self.assertIsInstance(prompt.messages[1], HumanMessagePromptTemplate)

    def test_context_to_additional_sources(self):
        expected = [
            {
                "url": "https://mock_source.com/mock.pdf",
                "snippet": "This is a mock document for testing purposes.",
                "page": 1,
            },
            {
                "url": "https://mock_source_2.com/mock.pdf",
                "snippet": "This is another mock document for testing purposes.",
                "page": 2,
            },
        ]
        for i, doc in enumerate(DOCS):
            self.assertEqual(convert_document_to_additional_source(doc), expected[i])

    def test_generator_fallback(self):
        def raise_error(msg: str):
            request = httpx.Request("GET", "/")
            response = httpx.Response(200, request=request)
            return RateLimitError(msg, response=response, body="")

        with patch(
            "app.generator.ChatOpenAI.invoke",
            side_effect=raise_error("OpenAI rate limit"),
        ), patch(
            "app.generator.ChatDeepSeek.invoke",
            side_effect=raise_error("DeepSeek rate limit"),
        ):
            generator = Generator()
            chain = {
                "question": RunnablePassthrough(),
                "context": RunnableLambda(lambda x: CONTEXT),
            } | generator()
            question = "Testing question for generator fallback"
            result = chain.invoke(question)
            self.assertEqual(result, LLM_FALLBACK)

    def test_conditional_prohibition(self):
        chain = {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(lambda x: CONTEXT),
        } | self.generator()
        question = "What is my name?"  # This question is not in the context
        result = chain.invoke(question)
        self.assertIn(
            NO_ANSWER_PROMPT.strip('"'),
            result["answer"],
        )
        self.assertEqual(result["citations"], [])
