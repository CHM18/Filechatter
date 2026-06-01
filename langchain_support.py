"""LangChain helpers for Filechatter.

This module provides prompt templating, structured output parsing, and a
simple retriever interface for the local RAG store.
"""

from __future__ import annotations

from typing import Any, Iterable

import requests
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document

import config
from rag_store import RagStore, SearchResult

_PROMPT_TEMPLATE = PromptTemplate(
    template=(
        "You are a helpful assistant. Answer the user question using the retrieved context when it is relevant."
        " If the context is insufficient, say that directly instead of inventing details.\n\n"
        "Retrieved context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Return a JSON object with these fields:\n"
        "- answer: the final answer to the user\n"
        "- sources: a JSON array of referenced source citations, each with source and chunk_index."
        " If no sources are used, return an empty list.\n\n"
        "{format_instructions}"
    ),
    input_variables=["context", "question", "format_instructions"],
)

_RESPONSE_SCHEMAS = [
    ResponseSchema(name="answer", description="The final answer to the user."),
    ResponseSchema(
        name="sources",
        description=(
            "A JSON array of referenced citations. Each item should include 'source' and 'chunk_index'. "
            "Return an empty list when the answer is not directly grounded in the provided context."
        ),
    ),
]

_OUTPUT_PARSER = StructuredOutputParser.from_response_schemas(_RESPONSE_SCHEMAS)

_SYSTEM_MESSAGE = (
    "You are a helpful assistant. Use the retrieved context when it is relevant, "
    "and avoid hallucinating facts."
)


def build_prompt(question: str, context: str) -> str:
    return _PROMPT_TEMPLATE.format(
        context=context or "No context available.",
        question=question,
        format_instructions=_OUTPUT_PARSER.get_format_instructions(),
    )


def parse_model_output(output: str) -> dict[str, Any]:
    return _OUTPUT_PARSER.parse(output)


def format_context(results: Iterable[Document]) -> str:
    blocks: list[str] = []
    for result in results:
        source = result.metadata.get("source", "<unknown>")
        chunk_index = result.metadata.get("chunk_index", "?")
        blocks.append(
            "\n".join(
                [
                    f"Source: {source}",
                    f"Chunk: {chunk_index}",
                    result.page_content,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


class LMStudioChatClient:
    def __init__(self, base_url: str, model: str, temperature: float, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    @property
    def system_message(self) -> str:
        return _SYSTEM_MESSAGE

    def create_chat_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": 0.9,
        }

    def request(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=self.create_chat_payload(prompt),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]


class RagRetriever(BaseRetriever):
    store: RagStore

    def __init__(self, store: RagStore) -> None:
        super().__init__(store=store)

    def get_relevant_documents(self, query: str) -> list[Document]:
        results = self.store.search(query)
        documents: list[Document] = []
        for result in results:
            metadata = dict(result.metadata)
            metadata.update(
                {
                    "chunk_id": result.chunk_id,
                    "source": result.source,
                    "score": result.score,
                    "rank": result.rank,
                }
            )
            documents.append(Document(page_content=result.content, metadata=metadata))
        return documents

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)
