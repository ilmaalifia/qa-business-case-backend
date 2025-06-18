import logging
import os
import sys

from app.state import AdditionalSource
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

TIMEOUT = 60  # seconds
MAX_RETRY = 3
RRF_CONSTANT = int(os.getenv("RRF_CONSTANT", "60"))
NUMBER_OF_CONTEXT_DOCS = int(
    os.getenv("NUMBER_OF_CONTEXT_DOCS", "5")
)  # number of context documents to be used in the prompt
TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.0")
)  # 0.0 (deterministic) - 1.0 (random)


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
        "url": doc.metadata.get("source"),
        "snippet": doc.page_content,
        "page": doc.metadata.get("page"),
    }


def get_bool_env(key: str, default: bool = True) -> bool:
    """Get boolean value from environment variable.

    Args:
        key: Environment variable name
        default: Default value if environment variable is not set

    Returns:
        Boolean value from environment variable
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "y")
