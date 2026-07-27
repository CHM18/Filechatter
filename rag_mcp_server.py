"""MCP server exposing Filechatter tools to MCP hosts (LM Studio, Claude Desktop, ...).

This is a thin stdio proxy: every tool call is forwarded over HTTP to the
Filechatter FastAPI server, which is the single process that owns the
SQLite/FAISS data. That means:

- Any number of MCP hosts can spawn this proxy concurrently without
  corrupting the index (each host launches its own copy).
- Startup is instant - no embedding model is loaded here.
- Permission settings changed in the web UI apply to the next tool call
  immediately, because enforcement happens centrally in the server.

The Filechatter server must be running (.\\rag.ps1 or python rag_server.py);
otherwise every tool returns a clear error message instead of failing.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

import config

# On Windows, Proactor + stdio pipes can emit noisy ConnectionResetError logs
# when MCP hosts terminate a session after their own timeout.
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SERVER_URL = os.getenv(
    "RAG_SERVER_URL",
    f"http://{config.RAG_SERVER_HOST}:{config.RAG_SERVER_PORT}",
).rstrip("/")

# Directory ingests run as background jobs server-side, so most calls return
# quickly. Single-file ingest of a large PDF can legitimately take a while.
DEFAULT_TIMEOUT = int(os.getenv("RAG_TOOL_TIMEOUT", "60"))
INGEST_FILE_TIMEOUT = int(os.getenv("RAG_INGEST_FILE_TIMEOUT", "300"))

INSTRUCTIONS = (
    "Use these tools to search a local document index and manage indexed sources. "
    "Prefer search_documents for grounded retrieval before answering questions. "
    "For large folders, use start_ingest_directory and poll get_ingest_status instead of "
    "a long-running ingest call. "
    f"All tools require the Filechatter server to be running at {SERVER_URL}."
)

mcp = FastMCP(
    "Filechatter",
    instructions=INSTRUCTIONS,
    json_response=True,
)


def _call(tool_name: str, arguments: dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> Any:
    """Forward a tool call to the Filechatter server."""
    try:
        response = requests.post(
            f"{SERVER_URL}/tools/{tool_name}",
            json={"arguments": arguments, "client": "mcp"},
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": (
                f"Filechatter server is not reachable at {SERVER_URL}. "
                "Start it first (.\\rag.ps1 or 'python rag_server.py'), then retry."
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": f"Filechatter server did not answer within {timeout}s for tool '{tool_name}'.",
        }

    if response.status_code != 200:
        detail: Any
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return {"status": "error", "message": f"Tool '{tool_name}' failed: {detail}"}

    payload = response.json()
    return payload.get("result", payload)


def search_documents(query: str, collection_name: str = "all") -> Any:
    """Search the local RAG index and return the most relevant chunks."""
    return _call("search_documents", {"query": query, "collection_name": collection_name})


def list_sources(collection_name: str = "all") -> Any:
    """List indexed sources and chunk counts in the local store."""
    return _call("list_sources", {"collection_name": collection_name})


def get_document_support() -> Any:
    """Report which document formats are currently ingest-ready and which optional dependencies are missing."""
    return _call("get_document_support", {})


def ingest_file(path: str, collection_name: str = "all") -> Any:
    """Index a single supported file into the local store."""
    return _call(
        "ingest_file",
        {"path": path, "collection_name": collection_name},
        timeout=INGEST_FILE_TIMEOUT,
    )


def start_ingest_directory(path: str, recursive: bool = True, collection_name: str = "all") -> Any:
    """Start indexing a directory in the background and return a job id for progress polling.

    Poll get_ingest_status(job_id) until status becomes 'completed' or 'failed'."""
    return _call(
        "start_ingest_directory",
        {"path": path, "recursive": recursive, "collection_name": collection_name},
    )


def get_ingest_status(job_id: str = "") -> Any:
    """Return the progress of a background ingest job. Omit job_id to inspect the active job."""
    return _call("get_ingest_status", {"job_id": job_id})


def get_ingest_log(job_id: str, tail_lines: int = 200) -> Any:
    """Return the tail of the ingest log file for a given job id."""
    return _call(
        "get_ingest_log",
        {"job_id": job_id, "tail_lines": tail_lines},
    )


def clear_index(confirm: bool = False, collection_name: str = "all", source: str = "") -> Any:
    """Clear the local index.

    - Set confirm=true to perform deletion.
    - Provide source to clear only one document.
    - Omit source to clear an entire collection (or all collections).
    """
    return _call(
        "clear_index",
        {"confirm": confirm, "collection_name": collection_name, "source": source},
    )


# All tools this proxy can expose. Which ones are actually registered depends
# on the server's permission settings (denied tools stay invisible).
ALL_TOOLS = {
    "search_documents": search_documents,
    "list_sources": list_sources,
    "get_document_support": get_document_support,
    "ingest_file": ingest_file,
    "start_ingest_directory": start_ingest_directory,
    "get_ingest_status": get_ingest_status,
    "get_ingest_log": get_ingest_log,
    "clear_index": clear_index,
}


def _fetch_visible_tool_names() -> set[str] | None:
    """Ask the server which tools are not denied. Returns None if the server
    is unreachable, meaning 'register everything and let the server enforce at
    call time' so tools are not silently missing when the proxy starts first."""
    try:
        response = requests.get(f"{SERVER_URL}/tools", params={"visible": "true"}, timeout=10)
        if response.status_code != 200:
            return None
        return {tool["name"] for tool in response.json().get("tools", [])}
    except requests.exceptions.RequestException:
        return None


def register_tools() -> list[str]:
    """Register the permitted tools with the MCP server. Denied tools are not
    advertised to the host. Visibility is decided at startup; changing it
    requires the host to reconnect/restart. Server-side enforcement is always
    live regardless of what is advertised here."""
    visible = _fetch_visible_tool_names()
    names = sorted(ALL_TOOLS) if visible is None else [n for n in ALL_TOOLS if n in visible]
    for name in names:
        mcp.add_tool(ALL_TOOLS[name])
    return names


register_tools()


if __name__ == "__main__":
    mcp.run()
