import logging
import sys

from app.state import AdditionalSource
from langchain_core.documents import Document

TIMEOUT = 20  # seconds
MAX_RETRY = 2
RRF_CONSTANT = 60
CONTEXT_DOCS = 5  # number of context documents to be used in the prompt


def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel("INFO")

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def convert_document_to_additional_source(doc: Document) -> AdditionalSource:
    """Convert a Document to an Additional Source."""
    return {
        "url": doc.metadata["source"],
        "snippet": doc.page_content,
        "page": doc.metadata.get("page"),
    }
