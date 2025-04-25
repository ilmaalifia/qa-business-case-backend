from typing import List, Optional

from langchain_core.documents import Document
from typing_extensions import Annotated, TypedDict


class Citation(TypedDict):
    """A citation used to justify the answer."""

    url: Annotated[
        str, ..., "The valid URL of the SPECIFIC source which justifies the answer."
    ]
    page: Annotated[
        Optional[int], ..., "Page number of the PDF source which justifies the answer."
    ]


class AdditionalSource(TypedDict):
    """A source that exists in context but is NOT USED to justifies the answer."""

    url: Annotated[
        str,
        ...,
        "The valid URL of the SPECIFIC source which is NOT USED to justifies the answer.",
    ]
    snippet: Annotated[
        str, ..., "The snippet of the source which is NOT USED to justifies the answer."
    ]
    page: Annotated[
        Optional[int],
        ...,
        "Page number of the PDF source which is NOT USED to justifies the answer.",
    ]


class InputState(TypedDict):
    question: Annotated[str, ..., "The question asked by the user."]


class ContextState(TypedDict):
    question: Annotated[str, ..., "The question asked by the user."]
    context: Annotated[
        List[Document],
        ...,
        "The high rank retrieved documents used as context used to answer the user question.",
    ]
    additional_sources: Annotated[
        Optional[List[Document]],
        [],
        "The low rank retrieved documents used as additional sources for user as reference.",
    ]


class OutputState(TypedDict):
    """Answer the user question based only on the given context and cite the sources used. Unused context should be listed as more sources."""

    answer: Annotated[
        str,
        ...,
        "The answer to the user question, which is based only on the given sources. If the given sources are insufficient, the answer should explain that the sources are insufficient.",
    ]
    citations: Annotated[
        List[Citation], ..., "The list of citations used to justify the answer."
    ]
    additional_sources: Annotated[
        Optional[List[AdditionalSource]],
        [],
        "The list of URLs of the sources which are NOT USED to justify the answer but exist in given context.",
    ]


def convert_document_to_additional_source(doc: Document) -> AdditionalSource:
    """Convert a Document to an AdditionalSource."""
    return {
        "url": doc.metadata["source"],
        "snippet": doc.page_content,
        "page": doc.metadata.get("page"),
    }
