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
    prompt = generator.get_prompt()
    llm = generator.get_llm()
    chain = (
        {"context": retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    question = "What is virtual power plant?"
    result = chain.invoke(question)
    logger.info(f"Query: {question}")
    logger.info(f"Result: {result}")
