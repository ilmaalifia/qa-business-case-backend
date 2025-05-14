import os
from typing import List

from app.state import OutputState
from app.utils import CONTEXT_DOCS, MAX_RETRY, TIMEOUT
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from openai import APIError, APITimeoutError, BadRequestError

load_dotenv()
PROHIBITION_PROMPT = (
    "I don't know the answer to that question due to insufficient context."
)


class Generator:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"You are a helpful assistant that answers a question based on the {CONTEXT_DOCS} context documents provided. If you don't know the answer or the context is insufficient, say {PROHIBITION_PROMPT}.",
                ),
                ("human", "Context: {context}\nQuestion: {question}"),
            ],
        )

        llm_provider = (os.getenv("LLM_PROVIDER", "")).upper()
        match llm_provider:
            case "OPENAI":
                self.llm = ChatOpenAI(
                    model="gpt-4o",
                    timeout=TIMEOUT,
                    max_retries=MAX_RETRY,
                    tags=[llm_provider.lower()],
                )
            case "DEEPSEEK":
                self.llm = ChatDeepSeek(
                    model="deepseek-chat",  # DeepSeek V3
                    timeout=TIMEOUT,
                    max_retries=MAX_RETRY,
                    tags=[llm_provider.lower()],
                )
            case _:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    # TODO: Manage doc positioning based on research
    def format_docs_as_context(self, docs: List[Document]):
        return (
            "\n"
            + "\n\n---\n\n".join(
                f"Source: {doc.metadata.get('source') or doc.metadata.get('Entry ID') or f'https://pubmed.ncbi.nlm.nih.gov/{doc.metadata.get('uid')}'}\nPage: {doc.metadata.get('page')}\nInformation: {doc.page_content}"
                for doc in docs
            )
            + "\n"
        )

    def __call__(self):
        return self.get_prompt() | self.get_llm()

    def get_prompt(self):
        return self.prompt

    def get_llm(self):
        return self.llm.with_structured_output(OutputState).with_fallbacks(
            self.__generator_fallback(),
            exceptions_to_handle=(APIError, APITimeoutError, BadRequestError),
        )

    def __generator_fallback(self):
        return [
            RunnableLambda(
                lambda x: {
                    "answer": "Unable to answer the question due to error. Please try again.",
                    "citations": [],
                    "additional_sources": [],
                }
            )
        ]
