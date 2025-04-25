import logging
import sys

TIMEOUT = 20  # seconds
MAX_RETRY = 2
RRF_CONSTANT = 60
CONTEXT_DOCS = 5


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
