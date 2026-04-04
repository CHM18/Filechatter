"""CLI for Filechatter RAG Server"""
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from document_loader import load_document, load_documents_from_directory
from rag_store import RagStore

app = typer.Typer(help="Filechatter CLI - Chat with your documents via LM Studio")
console = Console()

DEFAULT_SERVER = "http://localhost:8000"


def get_server_url(server: Optional[str] = None) -> str:
    """Get the server URL from parameter or environment"""
    return server or DEFAULT_SERVER


@app.command()
def chat(
    question: str = typer.Argument(..., help="Your question"),
    server: Optional[str] = typer.Option(
        None,
        "--server",
        "-s",
        help="RAG server URL (default: http://localhost:8000)"
    ),
    show_context: bool = typer.Option(
        True,
        "--show-context",
        help="Show retrieved context documents"
    ),
):
    """Chat with the RAG server"""
    url = get_server_url(server)
    
    with console.status("[bold cyan]Thinking..."):
        try:
            response = requests.post(
                f"{url}/chat",
                json={
                    "question": question,
                    "include_context": True
                },
                timeout=120
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            console.print(f"[red]Error: Cannot connect to server at {url}[/red]")
            raise typer.Exit(1)
        except requests.exceptions.Timeout:
            console.print("[red]Error: Request timed out[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    data = response.json()
    
    # Display the answer
    console.print(Panel(data["answer"], title="Answer", expand=False))
    
    # Display context if requested
    if show_context and data.get("context"):
        console.print("\n[bold cyan]Retrieved Context:[/bold cyan]")
        for i, chunk in enumerate(data["context"], 1):
            console.print(f"\n[yellow]Document {i}:[/yellow]")
            console.print(f"Source: {chunk['source']}")
            content = chunk["content"]
            console.print(content[:500] + ("..." if len(content) > 500 else ""))


@app.command()
def upload(
    file_path: Path = typer.Argument(..., help="Path to file to upload"),
    server: Optional[str] = typer.Option(
        None,
        "--server",
        "-s",
        help="RAG server URL"
    ),
):
    """Upload a document to the RAG database"""
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    try:
        loaded_document = load_document(file_path)
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(1)
    
    url = get_server_url(server)
    
    with console.status("[bold cyan]Uploading document..."):
        try:
            response = requests.post(
                f"{url}/upload",
                json={
                    "documents": [loaded_document.content],
                    "metadata": [loaded_document.metadata]
                },
                timeout=30
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            console.print(f"[red]Error: Cannot connect to server at {url}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    data = response.json()
    console.print(
        f"[green]✓ Uploaded {data['documents_uploaded']} document(s)[/green]"
    )
    console.print(f"  Chunks added: {data['chunks_uploaded']}")
    console.print(f"  Total chunks in database: {data['total_chunks']}")


@app.command()
def upload_dir(
    dir_path: Path = typer.Argument(..., help="Path to directory with supported files"),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Scan subdirectories recursively"
    ),
    server: Optional[str] = typer.Option(
        None,
        "--server",
        "-s",
        help="RAG server URL"
    ),
):
    """Upload all documents from a directory"""
    if not dir_path.is_dir():
        console.print(f"[red]Error: Directory not found: {dir_path}[/red]")
        raise typer.Exit(1)

    loaded_documents = load_documents_from_directory(dir_path, recursive=recursive)
    if not loaded_documents:
        console.print("[yellow]No supported files with readable content found[/yellow]")
        raise typer.Exit(0)

    documents = [document.content for document in loaded_documents]
    metadatas = [document.metadata for document in loaded_documents]
    
    url = get_server_url(server)
    
    with console.status("[bold cyan]Uploading documents..."):
        try:
            response = requests.post(
                f"{url}/upload",
                json={"documents": documents, "metadata": metadatas},
                timeout=60
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            console.print(f"[red]Error: Cannot connect to server at {url}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    data = response.json()
    console.print(
        f"[green]✓ Uploaded {data['documents_uploaded']} document(s)[/green]"
    )
    console.print(f"  Chunks added: {data['chunks_uploaded']}")
    console.print(f"  Total chunks in database: {data['total_chunks']}")


@app.command()
def status(
    server: Optional[str] = typer.Option(
        None,
        "--server",
        "-s",
        help="RAG server URL"
    ),
):
    """Check server status"""
    url = get_server_url(server)
    
    try:
        response = requests.get(f"{url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Create status table
        table = Table(title="Server Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")
        
        table.add_row("RAG Server", "[green]✓ Running[/green]")
        table.add_row(
            "LM Studio",
            "[green]✓ Connected[/green]" if data["lm_studio_connected"] else "[red]✗ Not connected[/red]"
        )
        table.add_row("Documents", str(data["documents_count"]))
        table.add_row("Chunks", str(data.get("chunks_count", 0)))
        
        console.print(table)
    
    except requests.exceptions.ConnectionError:
        console.print(f"[red]✗ Cannot connect to server at {url}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_docs(
    server: Optional[str] = typer.Option(
        None,
        "--server",
        "-s",
        help="RAG server URL"
    ),
):
    """List all documents in the database"""
    url = get_server_url(server)
    
    try:
        response = requests.get(f"{url}/documents", timeout=5)
        response.raise_for_status()
        data = response.json()

        console.print(f"[cyan]Total documents: {data['total_documents']}[/cyan]")
        console.print(f"[cyan]Total chunks: {data['total_chunks']}[/cyan]")
        if data.get("sources"):
            table = Table(title="Indexed Sources")
            table.add_column("Source", style="cyan")
            table.add_column("Chunks", style="magenta")
            table.add_column("Last Indexed", style="green")
            for source in data["sources"]:
                table.add_row(
                    source["source"],
                    str(source["chunk_count"]),
                    str(source["last_indexed_at"]),
                )
            console.print(table)
    
    except requests.exceptions.ConnectionError:
        console.print(f"[red]✗ Cannot connect to server at {url}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def dump_chunks(
    source: Optional[str] = typer.Argument(
        None,
        help="Exact source file name to inspect; omit to inspect the first chunks across all sources"
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of chunks to print"),
    offset: int = typer.Option(0, "--offset", help="Chunk offset for pagination"),
    server: Optional[str] = typer.Option(
        None,
        "--server",
        "-s",
        help="RAG server URL"
    ),
):
    """Print stored chunk content from the RAG index for debugging."""
    url = get_server_url(server)

    def print_chunks(chunks: list[dict]) -> None:
        if not chunks:
            console.print("[yellow]No chunks found for the requested source[/yellow]")
            raise typer.Exit(0)

        for chunk in chunks:
            header = (
                f"{chunk['source']} | chunk {chunk['chunk_index']} | "
                f"chars {chunk['start_char']}-{chunk['end_char']}"
            )
            console.print(Panel(chunk["content"], title=header, expand=False))

    try:
        response = requests.get(
            f"{url}/chunks",
            params={"source": source, "limit": limit, "offset": offset},
            timeout=10,
        )
        if response.status_code == 404:
            raise requests.exceptions.HTTPError("/chunks endpoint unavailable", response=response)
        response.raise_for_status()
        data = response.json()
        print_chunks(data.get("chunks", []))
        return
    except requests.exceptions.HTTPError as e:
        response = getattr(e, "response", None)
        if response is None or response.status_code != 404:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
        console.print("[yellow]Server does not expose /chunks yet; reading from local store instead[/yellow]")
    except requests.exceptions.ConnectionError:
        console.print("[yellow]Server unavailable; reading from local store instead[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    store = RagStore()
    try:
        chunks = store.get_chunks(source=source, limit=limit, offset=offset)
    finally:
        store.close()

    print_chunks(chunks)


@app.command()
def clear(
    server: Optional[str] = typer.Option(
        None,
        "--server",
        "-s",
        help="RAG server URL"
    ),
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation"
    ),
):
    """Clear all documents from the database"""
    if not confirm:
        if not typer.confirm("Are you sure you want to delete all documents?"):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)
    
    url = get_server_url(server)
    
    try:
        response = requests.delete(f"{url}/documents", timeout=5)
        response.raise_for_status()
        console.print("[green]✓ All documents cleared[/green]")
    
    except requests.exceptions.ConnectionError:
        console.print(f"[red]✗ Cannot connect to server at {url}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
