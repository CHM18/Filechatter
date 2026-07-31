// Minimal API helper for the Filechatter server (same origin).
"use strict";

const api = {
  async request(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      /* non-JSON response */
    }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : response.statusText;
      throw new Error(detail);
    }
    return payload;
  },

  health() {
    return this.request("/health");
  },

  settings() {
    return this.request("/settings");
  },

  updateSettings(changes) {
    return this.request("/settings", { method: "PUT", body: JSON.stringify({ changes }) });
  },

  setPermissions(permissions) {
    return this.request("/settings/permissions", {
      method: "PUT",
      body: JSON.stringify({ permissions }),
    });
  },

  listTools() {
    return this.request("/tools");
  },

  listCollections() {
    return this.request("/collections");
  },

  fileTypes() {
    return this.request("/file-types");
  },

  createCollection(body) {
    return this.request("/collections", { method: "POST", body: JSON.stringify(body) });
  },

  deleteCollection(name) {
    return this.request(`/collections/${encodeURIComponent(name)}?confirm=true`, { method: "DELETE" });
  },

  documentSupport() {
    return this.request("/tools/get_document_support", {
      method: "POST",
      body: JSON.stringify({ arguments: {} }),
    });
  },

  testEndpoint() {
    return this.request("/llm/test", { method: "POST" });
  },

  listModels() {
    return this.request("/llm/models");
  },

  listMemories() {
    return this.request("/memories");
  },

  updateMemory(id, body) {
    return this.request(`/memories/${id}`, { method: "PATCH", body: JSON.stringify(body) });
  },

  deleteMemory(id) {
    return this.request(`/memories/${id}`, { method: "DELETE" });
  },

  clearMemories() {
    return this.request("/memories?confirm=true", { method: "DELETE" });
  },

  // Stream chat events. `onEvent(evt)` is called per SSE event; resolves when the stream ends.
  async streamChat(body, onEvent) {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        detail = (await response.json()).detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8", { fatal: true });
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 2);
        if (chunk.startsWith("data:")) {
          onEvent(JSON.parse(chunk.slice(5).trim()));
        }
      }
    }
    buffer += decoder.decode();
    if (buffer.trim().startsWith("data:")) {
      onEvent(JSON.parse(buffer.trim().slice(5).trim()));
    }
  },
};

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2600);
}
