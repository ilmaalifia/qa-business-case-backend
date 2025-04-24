import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from utils import MAX_RETRY, TIMEOUT

cited_answer_schema = {
    "title": "cited_answer",
    "description": "Answer the user question based only on the given context, and cite the sources used.",
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the user question, which is based only on the given sources. If the given sources are insufficient, the answer should explain that the sources are insufficient.",
        },
        "citations": {
            "type": "array",
            "description": "The list of citations used to justify the answer.",
            "items": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The valid URL of the SPECIFIC source which justifies the answer.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "The page number of the PDF source which justifies the answer.",
                    },
                },
                "required": ["url"],
            },
        },
        "more_sources": {
            "type": "array",
            "description": "The list of URLs of the sources which are NOT USED to justify the answer but exist in given context.",
            "items": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The valid URL of the SPECIFIC source which is NOT USED to justifies the answer.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "The page number of the PDF source which is NOT USED to justifies the answer.",
                    },
                    "snippet": {
                        "type": "string",
                        "description": "The snippet of the source which is NOT USED to justifies the answer.",
                    },
                },
                "required": ["url", "snippet"],
            },
        },
    },
    "required": ["answer", "citations"],
}


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

    def get_prompt(self):
        return self.prompt

    def get_llm(self):
        return self.llm.with_structured_output(cited_answer_schema)
