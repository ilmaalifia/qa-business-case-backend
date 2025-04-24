from dotenv import load_dotenv
from generator import Generator
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
    )

    # question = "What is virtual power plant?"
    question = "What is the value proposition of vertical farming?"
    result = chain.invoke(question)
    logger.info(f"Question: {question}")
    logger.info(f"Result: {result}")
