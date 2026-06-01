#!/usr/bin/env python3
"""Test script to run ingest and capture logs."""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_mcp_server import start_ingest_directory, get_ingest_status, get_ingest_log

def main():
    test_dir = r"D:\CH_Dokumente\Science\Bücher_Physik"
    print(f"Starting ingest of: {test_dir}")
    print()
    
    # Start the ingest job
    result = start_ingest_directory(path=test_dir, recursive=True)
    job_id = result["job_id"]
    print(f"Job started: {job_id}")
    print(f"Total files to ingest: {result['total_files']}")
    print()
    
    # Poll until done
    while True:
        status = get_ingest_status(job_id=job_id)
        progress = status.get("progress_percent", 0)
        processed = status.get("processed_files", 0)
        total = status.get("total_files", 0)
        succeeded = status.get("succeeded_files", 0)
        failed = status.get("failed_files", 0)
        skipped = status.get("skipped_files", 0)
        
        print(f"[{progress}%] Processed: {processed}/{total} | Success: {succeeded} | Failed: {failed} | Skipped: {skipped}")
        
        if status["status"] in ("completed", "failed"):
            print(f"\nJob status: {status['status']}")
            break
        
        time.sleep(1)
    
    print()
    print("=" * 80)
    print("JOB SUMMARY")
    print("=" * 80)
    print(f"Status: {status['status']}")
    print(f"Total files: {total}")
    print(f"Processed: {processed}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Documents uploaded: {status.get('documents_uploaded', 0)}")
    print(f"Chunks uploaded: {status.get('chunks_uploaded', 0)}")
    
    if status.get("error"):
        print(f"\nGlobal error: {status['error']}")
    
    if status.get("recent_errors_full"):
        print(f"\nRecent errors ({len(status['recent_errors_full'])} total):")
        for err in status['recent_errors_full'][-5:]:
            print(f"  - {err[:200]}...")
    
    print()
    print("=" * 80)
    print("INGEST LOG (last 100 lines)")
    print("=" * 80)
    try:
        log_result = get_ingest_log(job_id=job_id, tail_lines=100)
        print(log_result["log"])
    except Exception as e:
        print(f"Error reading log: {e}")

if __name__ == "__main__":
    main()
