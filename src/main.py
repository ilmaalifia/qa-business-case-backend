from dotenv import load_dotenv
from generator import Generator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retriever import Retriever
from utils import setup_logger

logger = setup_logger(__name__)
load_dotenv()

if __name__ == "__main__":
    retriever = Retriever()
    generator = Generator()
    chain = (
        {"context": retriever(), "question": RunnablePassthrough()}
        | generator.get_prompt()
        | generator.get_llm()
        | StrOutputParser()
    )
    question = "What is virtual power plant?"
    result = chain.invoke(question)
    logger.info(f"Query: {question}")
    logger.info(f"Result: {result}")
