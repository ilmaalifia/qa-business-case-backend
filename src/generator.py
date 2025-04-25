import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from src.state import OutputState
from src.utils import MAX_RETRY, TIMEOUT

load_dotenv()


class Generator:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that answers question based on the context provided. If you don't know the answer or the context is insufficient, explain it clearly.
        Context:
        {context}
            
        Question: {question}"""
        )

        llm_provider = (os.getenv("LLM_PROVIDER", "")).upper()
        match llm_provider:
            case "OPENAI":
                self.llm = ChatOpenAI(
                    model="gpt-4o", timeout=TIMEOUT, max_retries=MAX_RETRY
                )
            case "DEEPSEEK":
                self.llm = ChatDeepSeek(
                    model="deepseek-chat",  # DeepSeek V3
                    timeout=TIMEOUT,
                    max_retries=MAX_RETRY,
                )
            case _:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    # TODO: Manage doc positioning based on research
    def format_docs_as_context(self, docs: List[Document]):
        return "\n\n---\n\n".join(
            f"Source: {doc.metadata['source']}\nPage: {doc.metadata.get("page")}\nInformation: {doc.page_content}"
            for doc in docs
        )

    def __call__(self):
        return self.prompt | self.llm.with_structured_output(OutputState)

    def get_prompt(self):
        return self.prompt

    def get_llm(self):
        return self.llm.with_structured_output(OutputState)


if __name__ == "__main__":
    """Test the system using LangChain Expression Language (LCEL)"""
    from langchain_core.runnables import RunnablePassthrough
    from src.retriever import Retriever
    from src.utils import setup_logger

    retriever = Retriever()
    generator = Generator()
    chain = {
        "context": retriever() | generator.format_docs_as_context,
        "question": RunnablePassthrough(),
    } | generator()

    question = "What is virtual power plant?"
    result = chain.invoke(question)
    print(f"Question: {question}")
    print(f"Result: {result}")
