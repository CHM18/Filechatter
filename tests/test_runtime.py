from __future__ import annotations

import threading

import runtime


def test_chat_turn_waits_for_queued_memory_extraction(data_env):
    queue = runtime.model_queue()
    extraction_started = threading.Event()
    release_extraction = threading.Event()
    chat_started = threading.Event()

    def extract():
        extraction_started.set()
        assert release_extraction.wait(timeout=1)

    def chat():
        with queue.chat_turn():
            chat_started.set()

    queue.enqueue_extraction(extract)
    assert extraction_started.wait(timeout=1)
    thread = threading.Thread(target=chat)
    thread.start()
    assert not chat_started.wait(timeout=0.05)

    release_extraction.set()
    assert chat_started.wait(timeout=1)
    thread.join(timeout=1)