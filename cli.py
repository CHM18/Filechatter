"""CLI for Filechatter RAG Server"""
import typer
import requests
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

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
        for i, doc in enumerate(data["context"], 1):
            console.print(f"\n[yellow]Document {i}:[/yellow]")
            console.print(doc[:500] + ("..." if len(doc) > 500 else ""))


@app.command()
def upload(
    file_path: Path = typer.Argument(..., help="Path to text file to upload"),
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
    
    # Read the file
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(1)
    
    url = get_server_url(server)
    
    with console.status("[bold cyan]Uploading document..."):
        try:
            response = requests.post(
                f"{url}/upload",
                json={
                    "documents": [content],
                    "metadata": [{"source": str(file_path.name)}]
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
    console.print(f"  Total documents in database: {data['total_documents']}")


@app.command()
def upload_dir(
    dir_path: Path = typer.Argument(..., help="Path to directory with text files"),
    pattern: str = typer.Option("*.txt", "--pattern", "-p", help="File pattern"),
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
    
    files = list(dir_path.glob(pattern))
    if not files:
        console.print(f"[yellow]No files matching '{pattern}' found[/yellow]")
        raise typer.Exit(0)
    
    documents = []
    metadatas = []
    
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            documents.append(content)
            metadatas.append({"source": file_path.name})
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")
    
    if not documents:
        console.print("[red]No documents could be loaded[/red]")
        raise typer.Exit(1)
    
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
    console.print(f"  Total documents in database: {data['total_documents']}")


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
    
    except requests.exceptions.ConnectionError:
        console.print(f"[red]✗ Cannot connect to server at {url}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


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
