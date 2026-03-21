# Filechatter - RAG Server with LM Studio

A modern RAG (Retrieval-Augmented Generation) server that integrates seamlessly with LM Studio for local language model inference. Chat with your documents using a local LLM.

## Features

- 🚀 **FastAPI Server** - High-performance API for RAG operations
- 📚 **Vector Database** - FAISS for efficient document retrieval
- 💬 **LM Studio Integration** - Use your local language models for responses
- 🎯 **Semantic Search** - Find relevant documents using embeddings
- 📁 **Flexible Document Upload** - Single files or entire directories
- 🖥️ **CLI Tool** - Easy command-line interface for interaction
- 🔌 **RESTful API** - Full HTTP API for integration with other apps

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

### Using the CLI

#### Upload a document:
```bash
# On Windows
venv\Scripts\python.exe -m cli upload documents/file.txt
# On macOS/Linux
python -m cli upload documents/file.txt
```

#### Upload all documents from a directory:
```bash
# On Windows
venv\Scripts\python.exe -m cli upload-dir ./documents --pattern "*.txt"
# On macOS/Linux
python -m cli upload-dir ./documents --pattern "*.txt"
```

#### Chat with your documents:
```bash
# On Windows
venv\Scripts\python.exe -m cli chat "What does the document say about X?"
# On macOS/Linux
python -m cli chat "What does the document say about X?"
```

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

#### Clear all documents:
```bash
# On Windows
venv\Scripts\python.exe -m cli clear --yes
# On macOS/Linux
python -m cli clear --yes
```

### API Endpoints

- `GET /health` - Server health check
- `POST /chat` - Send a question (with RAG context)
- `POST /upload` - Upload documents
- `GET /documents` - List document count
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
│   CLI / App     │
└────────┬────────┘
         │
    ┌────▼────────────────────────────┐
    │   FastAPI RAG Server            │
    │  (localhost:8000)               │
    └────┬────────────────┬───────────┘
         │                │
    ┌────▼──────┐    ┌────▼──────────────┐
    │   FAISS   │    │  LM Studio        │
    │  Vector DB│    │  (localhost:1234) │
    └───────────┘    └───────────────────┘
```

## Configuration

The server can be configured using environment variables. Copy `.env.example` to `.env` and modify the values:

- `LM_STUDIO_URL`: URL of your LM Studio server (default: http://localhost:1234)
- `LM_STUDIO_MODEL`: Model name to use (default: local-model)
- `CONTEXT_SIZE`: Number of documents to retrieve per query (default: 3)
- `LM_STUDIO_TIMEOUT`: Timeout for LM Studio requests in seconds (default: 60)
- `RAG_HOST`: Host for the RAG server (default: localhost)
- `RAG_PORT`: Port for the RAG server (default: 8000)
- `TEMPERATURE`: Temperature for LLM responses (default: 0.7)

## Project Structure

```
Filechatter/
├── rag_server.py       # Main RAG server (FastAPI)
├── cli.py              # CLI tool for interaction
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── venv/               # Virtual environment
```

## Troubleshooting

### "Cannot connect to LM Studio"
- Ensure LM Studio is running and a model is loaded
- Check the `LM_STUDIO_BASE_URL` in `rag_server.py`
- Default is `http://localhost:1234`

### "Connection refused on port 8000"
- Check if the server is running
- Try a different port by editing the server code

### Memory issues with large documents
- Consider splitting large files into smaller chunks
- Upload them individually or use `upload-dir` with filtered patterns

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Feel free to submit issues and pull requests.
