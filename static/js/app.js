// App shell: tab navigation and status panel.
"use strict";

const app = {
  async init() {
    document.querySelectorAll(".tab[data-panel]").forEach((tab) => {
      tab.addEventListener("click", () => this.showPanel(tab.dataset.panel));
    });

    await Promise.all([this.refreshStatus(), chatPanel.init()]);
    const chatRefresh = chatPanel.refresh();
    await Promise.all([
      collectionsPanel.init(),
      permissionsPanel.init(),
      memoryPanel.init(),
      chatRefresh,
    ]);
    await permissionsPanel.refresh();

    // Keep the status badge fresh.
    setInterval(() => this.refreshStatus(), 30000);
    // Refresh selectors while the server finishes background warmup.
    let startupRefreshes = 0;
    const refreshStartupData = async () => {
      if (startupRefreshes++ >= 12) return;
      // Chat refresh also loads the collection list for its selectors. Do not
      // re-render the Database tab here: model warmup can take several cycles,
      // and that would discard unsaved collection slider changes.
      await chatPanel.refresh();
      if (chatPanel.models.length && chatPanel.collections.length) return;
      setTimeout(refreshStartupData, 2500);
    };
    setTimeout(refreshStartupData, 1500);
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
        ["Read tools", settings.permissions.groups.read_tools || settings.permissions.groups.read],
        ["Ingest tools", settings.permissions.groups.ingest_tools || settings.permissions.groups.write],
        ["Collection administration", settings.permissions.groups.collection_tools || settings.permissions.groups.write],
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
