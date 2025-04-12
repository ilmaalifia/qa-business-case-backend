from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class Generator:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the context provided.

        Context: {context}

        Question: {question}"""
        )

        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def get_prompt(self):
        return self.prompt

    def get_llm(self):
        return self.llm
