from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langgraph.graph import END, START, StateGraph
from src.generator import Generator
from src.retriever import Retriever
from src.state import (
    ContextState,
    InputState,
    OutputState,
    convert_document_to_additional_source,
)
from src.utils import CONTEXT_DOCS

load_dotenv()

retriever = Retriever()
generator = Generator()


def retriever_node(input_state: InputState) -> ContextState:
    retrieved_docs = retriever().invoke(input_state["question"])
    return {
        "question": input_state["question"],
        "context": retrieved_docs[:CONTEXT_DOCS],
        "additional_sources": retrieved_docs[CONTEXT_DOCS:],
    }


def generator_node(context_state: ContextState) -> OutputState:
    prompt = generator.get_prompt().invoke(
        {
            "context": generator.format_docs_as_context(context_state["context"]),
            "question": context_state["question"],
        }
    )
    response = generator.get_llm().invoke(prompt)
    return {
        "answer": response["answer"],
        "citations": response["citations"],
        "additional_sources": response["additional_sources"]
        + [
            convert_document_to_additional_source(doc)
            for doc in context_state["additional_sources"]
        ],
    }


builder = StateGraph(InputState)
builder.add_node("retriever", retriever_node)
builder.add_node("generator", generator_node)
builder.add_edge(START, "retriever")
builder.add_edge("retriever", "generator")
builder.add_edge("generator", END)
graph = builder.compile()
graph.name = "QA System for Business Case"
