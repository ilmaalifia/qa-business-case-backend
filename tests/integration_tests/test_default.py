import json
import unittest

import pytest
from app.generator import Generator
from app.retriever import Retriever
from app.state import OutputState
from langchain_core.runnables import RunnablePassthrough
from pydantic import ValidationError


class TestDefault(unittest.TestCase):
    """Testing the default behaviour"""

    def test_chain(self):
        """Test the system using LangChain Expression Language (LCEL) and log the result for sanity check"""

        retriever = Retriever()
        generator = Generator()
        chain = {
            "context": retriever() | generator.format_docs_as_context,
            "question": RunnablePassthrough(),
        } | generator()

        question = "What is virtual power plant?"
        result = chain.invoke(question)
        try:
            OutputState(**result)
        except ValidationError:
            pytest.fail("ValidationError raised for valid input")
        finally:
            print(json.dumps(result))
