# Filechatter - Persistent Local RAG for LM Studio, Claude Desktop, and more

Filechatter is a local RAG backend. It chunks large text files, stores chunk metadata in SQLite, persists a FAISS index on disk, and exposes a FastAPI server with a web UI plus an MCP proxy so MCP hosts (LM Studio, Claude Desktop, ...) can search your documents during chat.

## Features

- 🚀 **FastAPI Server** - Single-writer server that owns all collection data
- 🌐 **Web UI** - Status dashboard and RAG database management at `http://localhost:8000/ui/`
- 💾 **Persistent Local Storage** - SQLite metadata plus FAISS vectors on disk
- 🗂️ **Managed Collections** - Create/delete RAG databases with description and allowed file types (`data/collections.json`)
- ✂️ **Chunked Ingestion** - Large files are split into retrieval-sized chunks before embedding
- 💬 **MCP Host Integration** - LM Studio, Claude Desktop, and other MCP hosts share one data backend safely
- 🎯 **Hybrid Search** - Combine semantic retrieval with exact keyword matching within a configured context budget
- 📁 **Flexible Document Upload** - Single files or entire directories, including subdirectories
- 🖥️ **CLI Tool** - Handy for ingestion and diagnostics
- 🔌 **RESTful API** - Upload, inspect, search, tool execution, and compatibility chat endpoints
- 🧠 **LangChain-backed chat** - Structured prompt template + output parser for grounded responses
- 🔧 **MCP Proxy** - Thin, instant-start stdio proxy; all tool calls execute centrally in the server

## Quick Start

### Prerequisites

- Python 3.9+ (restart VS Code after installation)
- LM Studio running on `http://localhost:1234` (or configure in code)
- A model loaded in LM Studio
- Clone the repository:
```bash
git clone <repo-url>
cd Filechatter
```

### Running the Server
1. including installation and setup
```powershell
.\rag.bat
```
2. after virtual environment was set up (second start, faster!)
```bash
.venv_filechatter\Scripts\activate.ps1
python ./rag_server.py
```

The server will start on `http://localhost:8000` and `rag.ps1` opens the web UI (`http://localhost:8000/ui/`) in your default browser once the server is healthy (suppress with `-noBrowser`). View the API docs at `http://localhost:8000/docs`.

Indexed data is persisted under `./data/<collection_name>/` by default. Collection metadata (description, allowed file types, `last_updated`) lives in `./data/collections.json`; existing collection directories are discovered and registered automatically on first start.

### Web UI

`http://localhost:8000/ui/` provides:

- **Chat** - talk to a local model with your documents as grounding; pick which databases are read from and written to, watch tool calls, and approve writes inline
- **Status** - server health, LM Studio connectivity, collection/document/chunk totals, active configuration
- **Databases** - list all RAG databases with live document/chunk counts, create new databases (name, description, allowed file types), delete databases
- **Permissions** - Ask/Allow/Deny control over the tools the model may call

### Chat

The Chat tab drives a tool-calling conversation against a **local, OpenAI-compatible
endpoint** (LM Studio today, Ollama or any `/v1/chat/completions` server next). Claude
Desktop and Microsoft Copilot are not chat targets here - they connect to Filechatter as
MCP hosts, and the web UI is their administration console.

Configure the endpoint in the Chat sidebar (provider, base URL, model) and press
**Reload models** (⟳). Then:

- **Read from / Write to** - tick which databases each message may read and write. Reads
  are scoped to the selected databases automatically (the model never has to name one).
- **Grounding** - the model calls `search_documents` against your read databases and cites
  what it used.
- **Writes** - respect the Permissions settings. With *Ask* (the default for writes), each
  ingest/clear pauses for your approval in the chat. When several write databases are
  selected, the model picks the best match by description/file-type; if it can't, you're
  asked to choose - all in one confirmation.
- **Sessions** - each browser tab is its own conversation; "New conversation" starts fresh.

### Tool permissions

The **Permissions** panel controls which tools the model may call. Read-only tools
(`search_documents`, `list_sources`, `get_document_support`, `get_ingest_status`,
`get_ingest_log`) and write tools (`ingest_file`, `start_ingest_directory`,
`clear_index`) each have a group setting:

- **Allow** - the tool runs without confirmation
- **Ask** - confirmation is required before it runs
- **Deny** - the tool is hidden from the model entirely and rejected if called anyway

Turn on **Advanced configuration** to override the decision per individual tool. The tools
are shown in two buckets (Read / Write), each headed by its group buttons; a tool without an
explicit override follows its group default (so a tool added in a future version is governed
by your group setting rather than silently allowed).

Clicking a **group button** applies that value to the whole group and clears any per-tool
overrides in it — exactly like simple mode. When a group contains diverging overrides, its
selected button is shown **faded** (a "mixed" state), including after you switch back to
simple mode, so stashed overrides stay visible. Clicking the group button again resets it to
a clean, uniform state.

These settings live in `data/settings.json` and apply to **both** the built-in chat
(arriving in M4) and MCP hosts. Enforcement is central and immediate: even if a host
still advertises a now-denied tool, the server rejects the call.

**Ask behavior differs by caller**, because only the browser can show Filechatter's own
confirmation dialog:

| Caller | Allow | Ask | Deny |
| --- | --- | --- | --- |
| Built-in web chat (M4) | runs | browser asks the user | hidden + rejected |
| MCP host (Claude Desktop, LM Studio) | runs | see `mcp_hosts.on_ask` below | hidden + rejected |

`mcp_hosts.on_ask` (Advanced → *MCP host behavior*) decides what an **Ask** tool does when
invoked from an MCP host:

- `host_confirm` (default) - run it; the host shows its own tool-use confirmation
- `elicit` - attempt MCP elicitation, otherwise fall back to running it
- `deny` - refuse the call with an explanatory message

> Deny/visibility changes reach MCP hosts only after the host reconnects or restarts
> (the proxy decides which tools to advertise at startup). Server-side enforcement of
> every rule is always live, so a stale advertisement cannot bypass a `deny`.

### Single-writer architecture

The FastAPI server is the **only** process that opens the SQLite/FAISS files. `rag_mcp_server.py` is a thin stdio proxy: every MCP tool call is forwarded over HTTP to `http://localhost:8000/tools/<name>`. Benefits:

- Multiple MCP hosts (LM Studio *and* Claude Desktop, each spawning their own proxy) can run concurrently without index corruption
- The MCP proxy starts instantly - no embedding model is loaded in the proxy process
- Writes to a collection are serialized through per-collection locks in the server

**Consequence: the Filechatter server must be running before MCP tools work.** If it is not, tools return a clear error message instead of failing.

## Optional: Manual steps
1. on windows install python
```bash
winget install Python.Python.3.12
```
2. Create a virtual environment:
```bash
py -m venv .venv_filechatter
# On Windows
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv_filechatter\Scripts\activate.ps1
# On macOS/Linux
source venv/bin/activate
```
3. Install dependencies:
```bash
# On Windows
winget install Rustlang.Rustup
# On macOS/Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# On Windows and macOS/Linux:
pip install -r requirements.txt
```

### Using LM Studio or Claude Desktop via MCP

Make sure the Filechatter server is running first (`.\rag.ps1`), then run the MCP proxy in a separate terminal (or let the MCP host launch it):

```bash
# On Windows
.venv_filechatter\Scripts\python.exe rag_mcp_server.py
# On macOS/Linux
python rag_mcp_server.py
```

Add it to LM Studio in `mcp.json` as a local program. Example Windows configuration:

```json
{
  "mcpServers": {
    "filechatter": {
      "command": "C:\\Users\\post\\Desktop\\Dateien_KI\\Filechatter\\.venv_filechatter\\Scripts\\python.exe",
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
- `get_document_support`
- `ingest_file`
- `start_ingest_directory`
- `get_ingest_status`
- `clear_index`

`ingest_file` and `start_ingest_directory` are version-aware for already indexed files:

- If the same `source` is ingested again with a newer file change timestamp, Filechatter replaces all old chunks for that source.
- If the incoming file change timestamp is older (or equal), Filechatter keeps the currently indexed version and skips replacement.
- Filechatter stores document-level metadata (source path/name, file changed date, and ingest date) so you can audit when a source was last updated in the RAG.

Most read tools accept a specific `collection_name` or `collection_name="all"`. Write tools also accept `collection_name="all"`; in that mode Filechatter auto-routes each file into the matching collection. If you pass a new explicit collection name, Filechatter creates it (registered in `collections.json` with default settings).

If a collection restricts its allowed file types (set in the web UI or via the API), `ingest_file` rejects non-matching files with a clear error, and `start_ingest_directory` skips them (reported as `skipped_by_type` in the job status).

If ingestion fails for a specific file type, call `get_document_support` first. It reports which formats are currently ingest-ready and which optional dependencies are missing.

For larger folders, prefer `start_ingest_directory` so the host does not sit in one long-running tool call and time out. Ingest jobs run to completion inside the server; if another job is already running, new jobs are queued and start automatically (job status `queued`).

Recommended workflow for large ingests:

1. Call `start_ingest_directory(path="...")`
2. Note the returned `job_id`
3. Poll `get_ingest_status(job_id="...")` until `status` becomes `completed` or `failed`

`get_ingest_status` returns `progress_percent`, `processed_files`, `total_files`, `current_file`, `chunks_uploaded`, `skipped_by_type`, and recent file-level errors. That is the practical way to surface progress in the host chat: show each poll result as a lightweight status update instead of waiting on a single blocking ingest call.

`list_sources` includes per-document timestamps to help with auditing updates:

- `file_changed_at`: file modification timestamp used for version checks
- `last_ingested_at`: when the source was last written to the RAG index

`clear_index` supports targeted and bulk deletes:

- Clear one document from one collection:
  - `clear_index(confirm=true, collection_name="documentation", source="C:\\path\\to\\file.docx")`
- Clear one document from all collections:
  - `clear_index(confirm=true, collection_name="all", source="C:\\path\\to\\file.docx")`
- Clear one full collection:
  - `clear_index(confirm=true, collection_name="documentation")`
- Clear all collections:
  - `clear_index(confirm=true, collection_name="all")`

### Using the CLI

#### Upload a document:
```bash
# On Windows
.venv_filechatter\Scripts\python.exe -m cli upload documents/file.txt
# On macOS/Linux
python -m cli upload documents/file.txt
```

#### Upload all supported documents from a directory recursively:
```bash
# On Windows
.venv_filechatter\Scripts\python.exe -m cli upload-dir ./documents
# On macOS/Linux
python -m cli upload-dir ./documents
```

Supported file types span several categories:

- **Images**: `jpg`, `jpeg`, `png`, `webp`, `gif`, `bmp`, `tif`, `tiff`, `avif`, `heic`
- **Office**: `pdf`, `docx`, `doc`, `xlsx`, `pptx`
- **Text**: `txt`, `md`, `markdown`, `rst`, `csv`, `tsv`, `log`
- **Web**: `html`, `htm`, `xml`, `css`, `vue`, `svelte`
- **Data & Config**: `json`, `yaml`, `toml`, `ini`, `cfg`, `conf`
- **Code**: `py`, `js`, `ts`, `java`, `c`, `cpp`, `go`, `rs`, `rb`, `php`, `sh`, `ps1`, `sql`, and many more

Image files are summarized through a vision model and stored together with selected EXIF fields;
the summary also includes file/folder-name context for better retrieval. Text, web, data, and
code files are read as plain UTF-8; HTML is stripped to text; Excel and PowerPoint need the
optional `openpyxl` / `python-pptx` dependencies (installed via
`requirements.txt`). When creating a database in the web UI you pick allowed types by category,
or switch to **Individual types** for per-extension control; leaving everything unselected
accepts all supported types.

For legacy `doc` files, Filechatter uses Microsoft Word automation on Windows when available. If Word or `pywin32` is unavailable, those files are skipped while the rest of the directory is still indexed.

#### Compatibility chat endpoint:
```bash
# On Windows
.venv_filechatter\Scripts\python.exe -m cli chat "What does the document say about X?"
# On macOS/Linux
python -m cli chat "What does the document say about X?"
```

This path still works, but the preferred workflow is to chat in LM Studio and let LM Studio call the MCP tools.

#### Check server status:
```bash
# On Windows
.venv_filechatter\Scripts\python.exe -m cli status
# On macOS/Linux
python -m cli status
```

#### List all documents:
```bash
# On Windows
.venv_filechatter\Scripts\python.exe -m cli list-docs
# On macOS/Linux
python -m cli list-docs
```

#### Dump stored chunk content for debugging:
```bash
# On Windows
.venv_filechatter\Scripts\python.exe -m cli dump-chunks "your-file.docx" --limit 3
# On macOS/Linux
python -m cli dump-chunks "your-file.docx" --limit 3
```

This prints the actual chunk text stored in the RAG index, which is useful when a document was uploaded but retrieval or summarization looks wrong.

#### Clear all documents:
```bash
# On Windows
.venv_filechatter\Scripts\python.exe -m cli clear --yes
# On macOS/Linux
python -m cli clear --yes
```

### API Endpoints

- `GET /health` - Server health check
- `GET /ui/` - Web frontend (status + database management)
- `GET /collections` - Collection metadata with live document/chunk counts
- `POST /collections` - Create a collection (`name`, `description`, `allowed_extensions`)
- `PATCH /collections/{name}` - Update description / allowed file types
- `DELETE /collections/{name}?confirm=true` - Delete a collection and its data
- `GET /tools` - List tools with read/write classification, parameter schemas, and resolved permission; `?visible=true` excludes denied tools
- `POST /tools/{name}` - Execute a tool, enforcing permissions (body: `{"arguments": {...}, "client": "api"|"mcp"}`); returns 403 when denied
- `GET /settings` / `PUT /settings` - Runtime settings (`data/settings.json`; secrets masked in responses)
- `PUT /settings/permissions` - Replace the permissions block (clears per-tool overrides)
- `GET /llm/models` / `POST /llm/test` - List models / test the configured chat endpoint
- `POST /chat/stream` - Built-in chat agent loop, streamed as Server-Sent Events
- `POST /search` - Retrieve relevant chunks only; accepts `collection_name`
- `POST /chat` - Compatibility endpoint that queries LM Studio directly; accepts `collection_name`
- `POST /upload` - Upload documents; accepts `collection_name`
- `GET /documents` - List indexed sources and chunk counts; accepts `collection_name`
- `GET /chunks` - Inspect stored chunks for debugging; accepts `collection_name`
- `DELETE /documents` - Clear documents; accepts `collection_name`

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

## Optional: Run Ollama locally via Docker

`ollama-docker/setup-ollama.sh` builds a small Ubuntu-based Docker image with
[Ollama](https://ollama.com) and starts it as a local, OpenAI-compatible chat endpoint you can
point Filechatter at (as an alternative to LM Studio). Run it from a bash shell (e.g. WSL2 on
Windows):

```bash
bash ollama-docker/setup-ollama.sh
```

The script:

- Builds the Docker image and (re)creates the `ollama-container` container, persisting models
  under `~/ollama`
- Detects an NVIDIA GPU on the host and, if found, installs the NVIDIA Container Toolkit if
  missing, generates the CDI spec needed for `--gpus all` (with WSL2-aware mode when running
  under WSL2), and verifies GPU access end-to-end before enabling it - falling back to CPU-only
  automatically if any step fails
- Waits for the Ollama API to become healthy, confirms the GPU is active inside the running
  container, then pulls the configured model (`nemotron-3-nano:4b` by default) and starts an
  interactive chat session

## Architecture

```
LM Studio ────stdio──┐
Claude Desktop stdio─┼─► rag_mcp_server.py (thin proxy, per host)
                     │        │ HTTP /tools/<name>
Browser (web UI) ────┼────────▼
CLI ─────────────────┴─► rag_server.py (localhost:8000)   ◄─ single writer
                          ├─ tool_registry.py  (read/write classified tools)
                          ├─ collections_manager.py (collections.json + stores)
                          ├─ ingest_jobs.py    (queued background ingest)
                          ├─ settings_manager.py (data/settings.json)
                          │
                ┌─────────┴─────────┐         ┌───────────────────┐
                │ SQLite + FAISS    │         │ LM Studio         │
                │ data/<collection> │         │ (localhost:1234)  │
                └───────────────────┘         └───────────────────┘
```

## Configuration

The server can be configured using environment variables. Copy `.env.example` to `.env` and modify the values:

- `LM_STUDIO_URL`: URL of your LM Studio server (default: http://localhost:1234)
- `LM_STUDIO_MODEL`: Model name to use (default: local-model)
- `IMAGE_VISION_BASE_URL`: OpenAI-compatible endpoint for image analysis (default: `LM_STUDIO_URL`)
- `IMAGE_VISION_MODEL`: Model used to summarize images (default: `LM_STUDIO_MODEL`)
- `IMAGE_VISION_API_KEY`: Optional API key for the vision endpoint
- `IMAGE_VISION_TIMEOUT`: Timeout for image-analysis requests in seconds
- `IMAGE_VISION_MAX_TOKENS`: Maximum output tokens for image summaries
- `IMAGE_EXIF_FIELDS`: Comma-separated EXIF tags to persist when present
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
├── rag_server.py           # FastAPI server: API, tool execution, web UI (single writer)
├── rag_mcp_server.py       # Thin MCP stdio proxy -> forwards tool calls to rag_server
├── tool_registry.py        # Shared tool definitions with read/write classification
├── permissions.py          # Ask/Allow/Deny resolution and enforcement
├── llm_providers.py        # OpenAI-compatible provider (streaming + tool calls)
├── chat_agent.py           # Chat agent loop: tool calling, scoping, confirmations
├── collections_manager.py  # Collection registry (data/collections.json) + store lifecycle
├── ingest_jobs.py          # Queued background directory-ingest jobs
├── settings_manager.py     # Runtime settings (data/settings.json)
├── runtime.py              # Process-wide singletons
├── rag_store.py            # SQLite + FAISS persistence and retrieval
├── document_loader.py      # Text extraction for docs + vision/EXIF summaries for images
├── langchain_support.py    # Prompt template, output parsing, LM Studio client
├── cli.py                  # CLI for upload, status, and diagnostics
├── static/                 # Web UI (vanilla JS, no build step)
├── tests/                  # pytest suite (isolated temp data dir, fake embedder)
├── requirements.txt        # Python dependencies
├── requirements-dev.txt    # Test dependencies (pytest, httpx)
└── README.md               # This file
```

## Running the tests

```bash
# Install dev dependencies once
.venv_filechatter\Scripts\python.exe -m pip install -r requirements-dev.txt
# Run the suite (no LM Studio or embedding model required)
.venv_filechatter\Scripts\python.exe -m pytest
```

Tests run against a temporary data directory with a deterministic fake embedding model - they never touch `./data` and need no network access.

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

### MCP tools return "Filechatter server is not reachable"
- The MCP proxy forwards all tool calls to the Filechatter server - start it first (`.\rag.ps1`)
- If the server runs on a non-default host/port, set `RAG_SERVER_URL` in the proxy's environment

### PDF or DOCX ingestion reports a missing dependency
- Call `get_document_support` in LM Studio to see reader readiness by file type
- Reinstall dependencies with `pip install -r requirements.txt`
- Restart the MCP server after installing new packages so it picks up the updated environment

### Some Word `.doc` files are skipped
- `.docx` is read directly in Python
- `.doc` requires Microsoft Word automation on Windows in this implementation
- If you need platform-independent `.doc` support, convert legacy `.doc` files to `.docx` first

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Feel free to submit issues and pull requests.

## To Do
- Replace custommade text splitter by llamaindex (or langchain?) text splitter (more complex but better quality). could solve this pblm as well: make it more robust: when many files are uploaded only some hits are returned for a key-word, not all
- make the collection selection wired through to the backend (currently only the chat via web-UI respects them)
- Add architectural details to source code project (dependency graph, classes...) so that the model gets a better overview
- make foto ingestion more meaningful, e.g. by multiple passes so that the llm learns who shown in the picture via meta data (and some guessing). Alternative: provide golden samples with persons taked with their names. Further: add a way to provide context information, e.g. birth dates, so that a birthday party can be assigned to a specific person...
- add the possibility to configure cron-jobs for ingestion runs per collection

