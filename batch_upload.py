#!/usr/bin/env python3
"""Batch upload script for Filechatter RAG Server"""
import requests
import os
import sys
from pathlib import Path

SERVER_URL = "http://localhost:8000"

def upload_documents(file_paths, server_url=SERVER_URL):
    """Upload multiple documents to the RAG server"""
    documents = []
    metadatas = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                documents.append(content)
                metadatas.append({"source": os.path.basename(file_path)})
                print(f"✓ Loaded: {file_path}")
        except Exception as e:
            print(f"✗ Failed to load {file_path}: {e}")

    if not documents:
        print("No documents to upload!")
        return

    try:
        response = requests.post(
            f"{server_url}/upload",
            json={"documents": documents, "metadata": metadatas},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        print(f"✓ Uploaded {result['documents_uploaded']} documents")
        print(f"  Total in database: {result['total_documents']}")
    except Exception as e:
        print(f"✗ Upload failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_upload.py <file1> <file2> ... <fileN>")
        print("Example: python batch_upload.py *.txt documents/*.md")
        sys.exit(1)

    file_paths = []
    for pattern in sys.argv[1:]:
        path = Path(pattern)
        if path.is_file():
            file_paths.append(str(path))
        elif "*" in pattern:
            # Handle glob patterns
            import glob
            file_paths.extend(glob.glob(pattern))
        else:
            print(f"Warning: {pattern} is not a file")

    if file_paths:
        print(f"Uploading {len(file_paths)} files...")
        upload_documents(file_paths)
    else:
        print("No files found!")