// App shell: tab navigation and status panel.
"use strict";

const app = {
  async init() {
    document.querySelectorAll(".tab[data-panel]").forEach((tab) => {
      tab.addEventListener("click", () => this.showPanel(tab.dataset.panel));
    });

    await Promise.all([
      this.refreshStatus(),
      collectionsPanel.init(),
      permissionsPanel.init(),
      chatPanel.init(),
      memoryPanel.init(),
    ]);
    await chatPanel.refresh(); // Chat is the default tab.

    // Keep the status badge fresh.
    setInterval(() => this.refreshStatus(), 30000);
  },

  showPanel(name) {
    document.querySelectorAll(".tab[data-panel]").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.panel === name);
    });
    document.querySelectorAll(".panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `panel-${name}`);
    });
    if (name === "chat") chatPanel.refresh();
    if (name === "memory") memoryPanel.refresh();
    if (name === "databases") collectionsPanel.refresh();
    if (name === "permissions") permissionsPanel.refresh();
  },

  async refreshStatus() {
    const badge = document.getElementById("server-badge");
    try {
      const health = await api.health();
      badge.textContent = "server online";
      badge.className = "badge badge-ok";
      setText("stat-server", "online", "ok");
      setText("stat-lmstudio", health.lm_studio_connected ? "connected" : "offline",
        health.lm_studio_connected ? "ok" : "err");
      setText("stat-collections", health.collections_count);
      setText("stat-documents", health.documents_count);
      setText("stat-chunks", health.chunks_count);
    } catch (error) {
      badge.textContent = "server unreachable";
      badge.className = "badge badge-err";
      setText("stat-server", "unreachable", "err");
      return;
    }

    try {
      const settings = await api.settings();
      const rows = [
        ["LLM provider", settings.llm.provider],
        ["LLM endpoint", settings.llm.base_url],
        ["Model", settings.llm.model],
        ["Temperature", settings.llm.temperature],
        ["Permissions mode", settings.permissions.mode],
        ["Read operations", settings.permissions.groups.read],
        ["Write operations", settings.permissions.groups.write],
      ];
      document.querySelector("#settings-table tbody").innerHTML = rows
        .map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td></tr>`)
        .join("");
    } catch (_) {
      /* settings endpoint unavailable */
    }
  },
};

function setText(id, value, cssClass) {
  const element = document.getElementById(id);
  element.textContent = value;
  element.className = "card-value" + (cssClass ? ` ${cssClass}` : "");
}

document.addEventListener("DOMContentLoaded", () => app.init());
