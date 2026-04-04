# Filechatter - Persistent Local RAG for LM Studio

Filechatter is a local RAG backend for LM Studio. It chunks large text files, stores chunk metadata in SQLite, persists a FAISS index on disk, and exposes both a FastAPI admin API and an MCP server so LM Studio can search your documents during chat.

## Features

- 🚀 **FastAPI Server** - High-performance API for RAG operations
- 💾 **Persistent Local Storage** - SQLite metadata plus FAISS vectors on disk
- ✂️ **Chunked Ingestion** - Large files are split into retrieval-sized chunks before embedding
- 💬 **LM Studio Integration** - Use your loaded LM Studio model with MCP tools
- 🎯 **Hybrid Search** - Combine semantic retrieval with exact keyword matching within a configured context budget
- 📁 **Flexible Document Upload** - Single files or entire directories, including subdirectories
- 🖥️ **CLI Tool** - Handy for ingestion and diagnostics
- 🔌 **RESTful API** - Upload, inspect, search, and compatibility chat endpoints
- 🔧 **MCP Server** - Lets LM Studio call the retriever directly from chat

## Quick Start

### Prerequisites

- Python 3.9+
- LM Studio running on `http://localhost:1234` (or configure in code)
- A model loaded in LM Studio

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd Filechatter
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Server

```bash
# On Windows
venv\Scripts\python.exe rag_server.py
# On macOS/Linux
python rag_server.py
```

The server will start on `http://localhost:8000`. View the API docs at `http://localhost:8000/docs`.

Indexed data is persisted under `./data` by default.

### Using LM Studio via MCP

Run the MCP server in a separate terminal:

```bash
# On Windows
venv\Scripts\python.exe rag_mcp_server.py
# On macOS/Linux
python rag_mcp_server.py
```

Add it to LM Studio in `mcp.json` as a local program. Example Windows configuration:

```json
{
  "mcpServers": {
    "filechatter": {
      "command": "C:\\Users\\post\\Desktop\\Dateien_KI\\Filechatter\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\post\\Desktop\\Dateien_KI\\Filechatter\\rag_mcp_server.py"
      ]
    }
  }
}
```

After that, LM Studio can use these tools during chat:

- `search_documents`
- `list_sources`
- `ingest_file`
- `ingest_directory`
- `start_ingest_directory`
- `get_ingest_status`
- `clear_index`

For small folders, `ingest_directory` is fine. For larger folders, prefer `start_ingest_directory` so LM Studio does not sit in one long-running tool call and time out.

Recommended LM Studio workflow for large ingests:

1. Call `start_ingest_directory(path="...")`
2. Note the returned `job_id`
3. Poll `get_ingest_status(job_id="...")` until `status` becomes `completed` or `failed`

`get_ingest_status` returns `progress_percent`, `processed_files`, `total_files`, `current_file`, `chunks_uploaded`, and recent file-level errors. That is the practical way to surface progress in LM Studio today: the chat can show each poll result as a lightweight status update instead of waiting on a single blocking ingest call.

### Using the CLI

#### Upload a document:
```bash
# On Windows
venv\Scripts\python.exe -m cli upload documents/file.txt
# On macOS/Linux
python -m cli upload documents/file.txt
```

#### Upload all supported documents from a directory recursively:
```bash
# On Windows
venv\Scripts\python.exe -m cli upload-dir ./documents
# On macOS/Linux
python -m cli upload-dir ./documents
```

Supported file types are `txt`, `pdf`, `docx`, and `doc`.

For legacy `doc` files, Filechatter uses Microsoft Word automation on Windows when available. If Word or `pywin32` is unavailable, those files are skipped while the rest of the directory is still indexed.

#### Compatibility chat endpoint:
```bash
# On Windows
venv\Scripts\python.exe -m cli chat "What does the document say about X?"
# On macOS/Linux
python -m cli chat "What does the document say about X?"
```

This path still works, but the preferred workflow is to chat in LM Studio and let LM Studio call the MCP tools.

#### Check server status:
```bash
# On Windows
venv\Scripts\python.exe -m cli status
# On macOS/Linux
python -m cli status
```

#### List all documents:
```bash
# On Windows
venv\Scripts\python.exe -m cli list-docs
# On macOS/Linux
python -m cli list-docs
```

#### Dump stored chunk content for debugging:
```bash
# On Windows
venv\Scripts\python.exe -m cli dump-chunks "your-file.docx" --limit 3
# On macOS/Linux
python -m cli dump-chunks "your-file.docx" --limit 3
```

This prints the actual chunk text stored in the RAG index, which is useful when a document was uploaded but retrieval or summarization looks wrong.

#### Clear all documents:
```bash
# On Windows
venv\Scripts\python.exe -m cli clear --yes
# On macOS/Linux
python -m cli clear --yes
```

### API Endpoints

- `GET /health` - Server health check
- `POST /search` - Retrieve relevant chunks only
- `POST /chat` - Compatibility endpoint that queries LM Studio directly
- `POST /upload` - Upload documents
- `GET /documents` - List indexed sources and chunk counts
- `GET /chunks` - Inspect stored chunks for debugging
- `DELETE /documents` - Clear all documents

### Example API Usage

```bash
# Chat with RAG
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics?",
    "include_context": true
  }'

# Search only
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics?",
    "include_context": true
  }'

# Upload documents
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["Document text here..."],
    "metadata": [{"source": "file.txt"}]
  }'
```

## Architecture

```
┌─────────────────┐
│ CLI / LM Studio │
└────────┬────────┘
         │
    ┌────▼────────────────────────────┐
  │ FastAPI Admin API / MCP Server  │
    │  (localhost:8000)               │
    └────┬────────────────┬───────────┘
         │                │
  ┌────▼──────────────┐    ┌────▼──────────────┐
  │ SQLite + FAISS    │    │ LM Studio         │
  │ local persistent  │    │ (localhost:1234)  │
  └───────────────────┘    └───────────────────┘
```

## Configuration

The server can be configured using environment variables. Copy `.env.example` to `.env` and modify the values:

- `LM_STUDIO_URL`: URL of your LM Studio server (default: http://localhost:1234)
- `LM_STUDIO_MODEL`: Model name to use (default: local-model)
- `DATA_DIR`: Directory for SQLite and FAISS persistence (default: ./data)
- `CHUNK_SIZE_CHARS`: Chunk size before embedding (default: 1800)
- `CHUNK_OVERLAP_CHARS`: Overlap between adjacent chunks (default: 250)
- `RETRIEVAL_K`: Number of chunks to fetch from FAISS before budgeting (default: 10)
- `KEYWORD_RETRIEVAL_K`: Number of keyword matches to fetch from SQLite FTS before fusion (default: 15)
- `MAX_CONTEXT_CHARS`: Maximum retrieved context passed to the model (default: 6000)
- `MAX_CHUNKS_PER_SOURCE`: Per-source cap during retrieval (default: 4)
- `SEMANTIC_WEIGHT`: Weight of embedding-based retrieval in hybrid ranking (default: 1.0)
- `KEYWORD_WEIGHT`: Weight of exact keyword retrieval in hybrid ranking (default: 2.0)
- `LM_STUDIO_TIMEOUT`: Timeout for LM Studio requests in seconds (default: 60)
- `RAG_HOST`: Host for the RAG server (default: localhost)
- `RAG_PORT`: Port for the RAG server (default: 8000)
- `TEMPERATURE`: Temperature for LLM responses (default: 0.7)

## Project Structure

```
Filechatter/
├── rag_server.py       # FastAPI admin API and compatibility chat endpoint
├── rag_mcp_server.py   # MCP server for LM Studio tool use
├── rag_store.py        # SQLite + FAISS persistence and retrieval
├── document_loader.py  # Text extraction for txt, pdf, docx, and doc
├── cli.py              # CLI for upload, status, and diagnostics
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Troubleshooting

### "Cannot connect to LM Studio"
- Ensure LM Studio is running and a model is loaded
- Check the `LM_STUDIO_BASE_URL` in `rag_server.py`
- Default is `http://localhost:1234`

### "Connection refused on port 8000"
- Check if the server is running
- Try a different port by editing the server code

### Retrieval is still too broad
- Reduce `MAX_CONTEXT_CHARS`
- Lower `MAX_CHUNKS_PER_SOURCE`
- Lower `CHUNK_SIZE_CHARS`

### LM Studio does not see the MCP server
- Ensure `rag_mcp_server.py` runs with the same virtual environment that has `mcp` installed
- Check the program entry in LM Studio `mcp.json`
- Restart LM Studio after editing `mcp.json`

### Some Word `.doc` files are skipped
- `.docx` is read directly in Python
- `.doc` requires Microsoft Word automation on Windows in this implementation
- If you need platform-independent `.doc` support, convert legacy `.doc` files to `.docx` first

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Feel free to submit issues and pull requests.

## To Do
- One-click-script to start venv and rag_server
- make it more robust: when many files are uploaded only some hits are returned for a key-word, not all
- include description of images into text (so that the LLM can better understand the context), especially images embedded in pdf and docx files
