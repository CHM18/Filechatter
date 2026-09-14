// Chat panel: endpoint config, read/write database selection, streaming
// conversation with tool cards and confirmation prompts.
"use strict";

const PROVIDER_DEFAULTS = {
  lmstudio: "http://localhost:1234",
  ollama: "http://localhost:11434",
  openai_compat: "",
};

const chatPanel = {
  settings: null,
  collections: [],
  sessionId: null,
  busy: false,
  currentTurn: null, // {contentEl, toolCards: {id: el}}
  modelsLoadedKey: null,
  modelsLoadedAt: 0,

  async init() {
    document.getElementById("chat-form").addEventListener("submit", (e) => {
      e.preventDefault();
      this.send();
    });
    document.getElementById("chat-text").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.send();
      }
    });
    document.getElementById("chat-provider").addEventListener("change", (e) => {
      const base = PROVIDER_DEFAULTS[e.target.value];
      if (base) document.getElementById("chat-base-url").value = base;
      this.saveEndpoint();
    });
    document.getElementById("chat-base-url").addEventListener("change", () => this.saveEndpoint());
    document.getElementById("chat-model").addEventListener("change", () => this.saveEndpoint());
    document.getElementById("chat-refresh-models").addEventListener("click", () => this.loadModels(true));
    document.getElementById("chat-test").addEventListener("click", () => this.testConnection());
    document.getElementById("chat-reset").addEventListener("click", () => this.resetConversation());
  },

  async refresh() {
    try {
      [this.settings, this.collections] = await Promise.all([
        api.settings(),
        api.listCollections().then((p) => p.collections || []),
      ]);
      this.renderEndpoint();
      this.renderDbSelectors();
      await this.maybeLoadModels();
    } catch (error) {
      showToast(`Failed to load chat config: ${error.message}`);
    }
  },

  // ---------- Endpoint ----------

  renderEndpoint() {
    const llm = this.settings.llm;
    document.getElementById("chat-provider").value = llm.provider in PROVIDER_DEFAULTS ? llm.provider : "openai_compat";
    document.getElementById("chat-base-url").value = llm.base_url || "";
    const modelSelect = document.getElementById("chat-model");
    modelSelect.innerHTML = `<option value="${escapeHtml(llm.model)}">${escapeHtml(llm.model)}</option>`;
  },

  currentModelKey() {
    const provider = document.getElementById("chat-provider").value;
    const baseUrl = document.getElementById("chat-base-url").value.trim();
    return `${provider}|${baseUrl}`;
  },

  async maybeLoadModels() {
    const key = this.currentModelKey();
    const stale = Date.now() - this.modelsLoadedAt > 60_000;
    if (this.modelsLoadedKey !== key || stale || !document.getElementById("chat-model").options.length) {
      await this.loadModels(false);
    }
  },

  async saveEndpoint() {
    const provider = document.getElementById("chat-provider").value;
    const baseInput = document.getElementById("chat-base-url");
    let baseUrl = baseInput.value.trim();
    if (!baseUrl && PROVIDER_DEFAULTS[provider]) {
      baseUrl = PROVIDER_DEFAULTS[provider];
      baseInput.value = baseUrl;
    }
    if (baseUrl && !/^https?:\/\//i.test(baseUrl)) {
      baseUrl = `http://${baseUrl.replace(/^\/+/, "")}`;
      baseInput.value = baseUrl;
    }

    const changes = {
      llm: {
        provider,
        base_url: baseUrl,
        model: document.getElementById("chat-model").value,
      },
    };
    try {
      this.settings = await api.updateSettings(changes);
      this.modelsLoadedKey = null;
      app.refreshStatus();
    } catch (error) {
      showToast(`Could not save endpoint: ${error.message}`);
    }
  },

  async loadModels(force = false) {
    const status = document.getElementById("chat-endpoint-status");
    status.textContent = force ? "Reloading models…" : "Loading models…";
    status.className = "endpoint-status";
    try {
      await this.saveEndpoint(); // persist base_url first so the server queries the right host
      const models = (await api.listModels(force)).models || [];
      const select = document.getElementById("chat-model");
      const current = this.settings.llm.model;
      select.innerHTML = models.map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
      if (models.includes(current)) select.value = current;
      else if (models.length) this.saveEndpoint();
      this.modelsLoadedKey = this.currentModelKey();
      this.modelsLoadedAt = Date.now();
      status.textContent = `${models.length} model(s) available`;
      status.classList.add("ok");
    } catch (error) {
      status.textContent = error.message;
      status.classList.add("err");
    }
  },

  async testConnection() {
    const status = document.getElementById("chat-endpoint-status");
    status.textContent = "Testing…";
    status.className = "endpoint-status";
    try {
      await this.saveEndpoint();
      const result = await api.testEndpoint();
      if (result.ok) {
        status.textContent = `Connected — ${result.model_count} model(s)`;
        status.classList.add("ok");
      } else {
        status.textContent = result.error;
        status.classList.add("err");
      }
    } catch (error) {
      status.textContent = error.message;
      status.classList.add("err");
    }
  },

  // ---------- Database selectors ----------

  renderDbSelectors() {
    const defaults = this.settings.chat_defaults || { read_collections: [], write_collections: [] };
    const names = this.collections.map((c) => c.name);
    const readSel = defaults.read_collections.filter((n) => names.includes(n));
    const writeSel = defaults.write_collections.filter((n) => names.includes(n));
    // Default read = all collections when nothing stored yet.
    const initialRead = readSel.length || defaults.read_collections.length ? readSel : names;

    this.buildChecklist("chat-read-dbs", "read", initialRead);
    this.buildChecklist("chat-write-dbs", "write", writeSel);
  },

  buildChecklist(containerId, kind, selected) {
    const container = document.getElementById(containerId);
    if (!this.collections.length) {
      container.innerHTML = '<div class="hint">No databases yet.</div>';
      return;
    }
    container.innerHTML = this.collections
      .map((c) => {
        const checked = selected.includes(c.name) ? "checked" : "";
        return `<label class="db-item"><input type="checkbox" data-kind="${kind}" value="${escapeHtml(c.name)}" ${checked}> ${escapeHtml(c.name)}</label>`;
      })
      .join("");
    container.querySelectorAll("input").forEach((input) =>
      input.addEventListener("change", () => this.saveDbDefaults())
    );
  },

  selectedDbs(kind) {
    return Array.from(
      document.querySelectorAll(`#panel-chat input[data-kind="${kind}"]:checked`)
    ).map((input) => input.value);
  },

  async saveDbDefaults() {
    try {
      await api.updateSettings({
        chat_defaults: {
          read_collections: this.selectedDbs("read"),
          write_collections: this.selectedDbs("write"),
        },
      });
    } catch (_) {
      /* non-critical */
    }
  },

  // ---------- Conversation ----------

  resetConversation() {
    this.sessionId = null;
    this.currentTurn = null;
    document.getElementById("chat-messages").innerHTML =
      '<div class="chat-empty">New conversation started.</div>';
  },

  addBubble(role, text) {
    const messages = document.getElementById("chat-messages");
    messages.querySelector(".chat-empty")?.remove();
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;
    bubble.textContent = text;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
  },

  startAssistantTurn() {
    const messages = document.getElementById("chat-messages");
    const turn = document.createElement("div");
    turn.className = "chat-bubble assistant";
    const tools = document.createElement("div");
    tools.className = "tool-cards";
    const content = document.createElement("div");
    content.className = "assistant-text";
    const status = document.createElement("div");
    status.className = "assistant-status";
    status.textContent = "Processing prompt…";
    turn.append(tools, status, content);
    messages.appendChild(turn);
    messages.scrollTop = messages.scrollHeight;
    this.currentTurn = {
      turnEl: turn,
      toolsEl: tools,
      statusEl: status,
      contentEl: content,
      toolCards: {},
      raw: "",
      sources: new Map(),
      usage: null,
    };
  },

  async send() {
    if (this.busy) return;
    const textarea = document.getElementById("chat-text");
    const message = textarea.value.trim();
    if (!message) return;
    if (!this.selectedDbs("read").length && !this.selectedDbs("write").length) {
      showToast("Select at least one database to read from or write to.");
    }
    textarea.value = "";
    this.addBubble("user", message);
    await this.runStream({ message });
  },

  async runStream(extra) {
    this.busy = true;
    this.setSending(true);
    this.startAssistantTurn();
    this.currentTurn.statusEl.textContent = "Processing prompt…";
    const body = {
      session_id: this.sessionId,
      read_collections: this.selectedDbs("read"),
      write_collections: this.selectedDbs("write"),
      ...extra,
    };
    try {
      await api.streamChat(body, (event) => this.handleEvent(event));
    } catch (error) {
      this.currentTurn.contentEl.innerHTML += `<div class="chat-error">Error: ${escapeHtml(error.message)}</div>`;
    } finally {
      this.busy = false;
      this.setSending(false);
    }
  },

  handleEvent(event) {
    const turn = this.currentTurn;
    switch (event.type) {
      case "session":
        this.sessionId = event.session_id;
        break;
      case "token":
        turn.raw += event.text;
        turn.contentEl.textContent = turn.raw;
        turn.statusEl.textContent = "Generating response…";
        this.scroll();
        break;
      case "usage":
        turn.usage = event.usage || null;
        turn.statusEl.textContent = this.formatUsage(turn.usage);
        break;
      case "tool_call":
        this.renderToolCard(event);
        break;
      case "tool_result":
        this.updateToolCard(event);
        this.collectSources(event);
        break;
      case "need_confirmation":
        this.renderConfirmations(event.pending);
        break;
      case "done":
        if (!turn.raw.trim() && event.content) turn.raw = event.content;
        if (turn.raw.trim()) {
          turn.contentEl.innerHTML = renderMarkdown(turn.raw);
        } else {
          turn.contentEl.innerHTML = '<div class="chat-error">No answer returned.</div>';
          turn.statusEl.textContent = "No answer returned.";
        }
        this.renderSources();
        if (turn.usage) turn.statusEl.textContent = this.formatUsage(turn.usage);
        break;
      case "error":
        turn.statusEl.textContent = "Failed";
        turn.contentEl.innerHTML += `<div class="chat-error">${escapeHtml(event.message)}</div>`;
        break;
    }
  },

  formatUsage(usage) {
    if (!usage) return "Request complete";
    const pieces = [];
    if (usage.prompt_tokens != null) pieces.push(`prompt ${usage.prompt_tokens}`);
    if (usage.completion_tokens != null) pieces.push(`completion ${usage.completion_tokens}`);
    if (usage.total_tokens != null) pieces.push(`tokens used: ${usage.total_tokens}`);
    return pieces.length ? pieces.join(" · ") : "Request complete";
  },

  renderToolCard(event) {
    const card = document.createElement("div");
    card.className = "tool-card running";
    card.innerHTML =
      `<div class="tool-card-head">🔧 <b>${escapeHtml(event.name)}</b> <span class="tool-status">running…</span></div>` +
      `<pre class="tool-args">${escapeHtml(JSON.stringify(event.arguments))}</pre>`;
    this.currentTurn.toolsEl.appendChild(card);
    this.currentTurn.toolCards[event.id] = card;
    this.scroll();
  },

  updateToolCard(event) {
    const card = this.currentTurn.toolCards[event.id];
    if (!card) return;
    card.classList.remove("running");
    card.classList.add(event.ok ? "ok" : "fail");
    const status = card.querySelector(".tool-status");
    let summary = "done";
    let note = "";
    if (Array.isArray(event.result)) {
      summary = event.result.length ? `${event.result.length} result(s)` : "empty result";
      if (!event.result.length) note = "No results returned.";
    } else if (event.result && typeof event.result === "object") {
      summary = Object.keys(event.result).length ? "done" : "empty result";
      if (summary === "empty result") note = "No result payload returned.";
    } else if (event.result === null || event.result === undefined || event.result === "") {
      summary = "empty result";
      note = "No result returned.";
    }
    if (!event.ok) {
      summary = "failed";
      note = typeof event.result === "string" && event.result ? event.result : "Tool call failed or timed out.";
    }
    status.textContent = summary;
    card.title = typeof event.result === "string" ? event.result : JSON.stringify(event.result);
    card.querySelector(".tool-note")?.remove();
    if (note) {
      const info = document.createElement("div");
      info.className = "tool-note";
      info.textContent = note;
      card.appendChild(info);
    }
  },

  renderConfirmations(pending) {
    const card = document.createElement("div");
    card.className = "confirm-card";
    const decisions = {};

    pending.forEach((item) => {
      const block = document.createElement("div");
      block.className = "confirm-item";
      const kind = item.options ? "write (choose database)" : "action";
      block.innerHTML =
        `<div class="confirm-head">The model wants to run <b>${escapeHtml(item.name)}</b> — ${kind}</div>` +
        `<pre class="tool-args">${escapeHtml(JSON.stringify(item.arguments))}</pre>`;

      let collectionSelect = null;
      if (item.options) {
        collectionSelect = document.createElement("select");
        collectionSelect.className = "confirm-select";
        collectionSelect.innerHTML = item.options
          .map((o) => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`)
          .join("");
        const label = document.createElement("label");
        label.className = "side-field";
        label.textContent = "Target database";
        label.appendChild(collectionSelect);
        block.appendChild(label);
      }
      block.dataset.id = item.id;
      block._select = collectionSelect;
      card.appendChild(block);
    });

    const actions = document.createElement("div");
    actions.className = "confirm-actions";
    const allow = document.createElement("button");
    allow.className = "btn btn-primary";
    allow.textContent = "Allow";
    const deny = document.createElement("button");
    deny.className = "btn";
    deny.textContent = "Deny";
    actions.append(deny, allow);
    card.appendChild(actions);

    const finish = (approved) => {
      card.querySelectorAll("button").forEach((b) => (b.disabled = true));
      pending.forEach((item) => {
        const block = card.querySelector(`.confirm-item[data-id="${item.id}"]`);
        decisions[item.id] = { approved };
        if (approved && block._select) decisions[item.id].collection = block._select.value;
      });
      card.classList.add(approved ? "resolved-allow" : "resolved-deny");
      this.runStream({ decisions });
    };
    allow.addEventListener("click", () => finish(true));
    deny.addEventListener("click", () => finish(false));

    this.currentTurn.toolsEl.appendChild(card);
    this.scroll();
  },

  collectSources(event) {
    if (event.name !== "search_documents" || !Array.isArray(event.result)) return;
    event.result.forEach((hit) => {
      if (hit && hit.source) {
        this.currentTurn.sources.set(hit.source, hit.collection_name || "");
      }
    });
  },

  renderSources() {
    const sources = this.currentTurn.sources;
    if (!sources.size) return;
    const footer = document.createElement("div");
    footer.className = "chat-sources";
    const items = Array.from(sources.entries())
      .map(([source, collection]) => {
        const name = source.split(/[\\/]/).pop() || source;
        const tag = collection ? `<span class="src-tag">${escapeHtml(collection)}</span>` : "";
        return `<li title="${escapeHtml(source)}">${tag}${escapeHtml(name)}</li>`;
      })
      .join("");
    footer.innerHTML = `<div class="src-title">Sources</div><ul>${items}</ul>`;
    this.currentTurn.turnEl.appendChild(footer);
    this.scroll();
  },

  setSending(sending) {
    document.getElementById("chat-send").disabled = sending;
    document.getElementById("chat-send").textContent = sending ? "…" : "Send";
  },

  scroll() {
    const messages = document.getElementById("chat-messages");
    messages.scrollTop = messages.scrollHeight;
  },
};

// Minimal, safe markdown: escape everything, then apply a few inline formats.
// Code fences are extracted first and restored escaped so nothing executes.
function renderMarkdown(text) {
  const fences = [];
  let s = String(text).replace(/```([\s\S]*?)```/g, (m, code) => {
    fences.push(code);
    return `@@FENCE${fences.length - 1}@@`;
  });
  s = escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/^#{1,6}\s*(.*)$/gm, "<b>$1</b>")
    .replace(/\n/g, "<br>");
  return s.replace(/@@FENCE(\d+)@@/g, (m, i) => `<pre class="md-code">${escapeHtml(fences[i])}</pre>`);
}
