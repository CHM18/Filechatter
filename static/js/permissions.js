// Global tool access controls shown in the Databases tab.
"use strict";

const PERM_VALUES = ["ask", "allow", "deny"];
const PERM_LABELS = { ask: "Ask", allow: "Allow", deny: "Deny" };
const PERM_GROUPS = [
  { key: "read_tools", label: "Read tools", hint: "search, list, document support" },
  { key: "ingest_tools", label: "Ingest tools", hint: "indexing and ingest status" },
  { key: "collection_tools", label: "Collection administration", hint: "create, clear, delete" },
];

const permissionsPanel = {
  settings: null,
  tools: [],

  async init() {
    document.getElementById("global-on-ask-select").addEventListener("change", (event) => {
      this.saveOnAsk(event.target.value);
    });
  },

  async refresh() {
    try {
      const [settings, toolsPayload] = await Promise.all([api.settings(), api.listTools()]);
      this.settings = settings;
      this.tools = toolsPayload.tools || [];
      this.render();
    } catch (error) {
      showToast(`Failed to load global tool access: ${error.message}`);
    }
  },

  groupValue(key) {
    const groups = this.settings.permissions.groups || {};
    if (groups[key]) return groups[key];
    if (key === "read_tools") return groups.read || "allow";
    return groups.write || "ask";
  },

  render() {
    if (!this.settings) return;
    const container = document.getElementById("global-perm-container");
    container.innerHTML = "";
    PERM_GROUPS.forEach((group) => {
      const cell = document.createElement("div");
      cell.className = "global-perm-cell";
      const title = document.createElement("div");
      title.className = "global-perm-title";
      title.textContent = group.label;
      const hint = document.createElement("span");
      hint.className = "hint";
      hint.textContent = group.hint;
      title.appendChild(hint);
      cell.appendChild(title);
      const selector = document.createElement("div");
      selector.className = "seg";
      PERM_VALUES.forEach((value) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "seg-btn";
        button.textContent = PERM_LABELS[value];
        if (value === this.groupValue(group.key)) button.classList.add("sel");
        button.addEventListener("click", () => this.setGroup(group.key, value));
        selector.appendChild(button);
      });
      cell.appendChild(selector);
      container.appendChild(cell);
    });
    document.getElementById("global-on-ask-select").value = this.settings.mcp_hosts.on_ask;
  },

  async setGroup(key, value) {
    const groups = {
      read_tools: this.groupValue("read_tools"),
      ingest_tools: this.groupValue("ingest_tools"),
      collection_tools: this.groupValue("collection_tools"),
      [key]: value,
    };
    try {
      this.settings = await api.setPermissions({ mode: "simple", groups, tools: {} });
      await this.refresh();
      collectionsPanel.refresh();
      app.refreshStatus();
    } catch (error) {
      showToast(`Could not save global tool access: ${error.message}`);
    }
  },

  async saveOnAsk(value) {
    try {
      this.settings = await api.updateSettings({ mcp_hosts: { on_ask: value } });
    } catch (error) {
      showToast(`Could not save MCP host behavior: ${error.message}`);
    }
  },
};
