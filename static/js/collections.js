// Databases panel: list, create, delete collections.
"use strict";

const collectionsPanel = {
  catalog: [], // [{category, extensions:[{ext,available,dependency}]}]
  selectedExts: new Set(),
  pendingDelete: null,

  async init() {
    document.getElementById("btn-new-collection").addEventListener("click", () => this.openCreateDialog());
    document.getElementById("btn-apply-all").addEventListener("click", () => this.saveAll());
    document.getElementById("btn-create-cancel").addEventListener("click", () => {
      document.getElementById("dialog-create").close();
    });
    document.getElementById("btn-delete-cancel").addEventListener("click", () => {
      document.getElementById("dialog-delete").close();
    });
    document.getElementById("form-create").addEventListener("submit", (event) => this.submitCreate(event));
    document.getElementById("form-delete").addEventListener("submit", (event) => this.submitDelete(event));
    document.getElementById("ext-advanced").addEventListener("change", () => this.renderExtSelector());
    [
      ["create-semantic-weight", "create-semantic-value"],
      ["create-keyword-weight", "create-keyword-value"],
      ["create-metadata-weight", "create-metadata-value"],
    ].forEach(([rangeId, valueId]) => {
      const range = document.getElementById(rangeId);
      const output = document.getElementById(valueId);
      if (range && output) {
        range.addEventListener("input", () => {
          output.textContent = Number(range.value).toFixed(1);
        });
      }
    });

    // Fetch the file-type catalog (categories + per-extension availability).
    try {
      this.catalog = (await api.fileTypes()).categories || [];
    } catch (_) {
      this.catalog = [];
    }

    await this.refresh();
  },

  async refresh() {
    const container = document.getElementById("collections-list");
    try {
      const payload = await api.listCollections();
      this.render(payload.collections || []);
    } catch (error) {
      container.innerHTML = `<div class="empty-state">Failed to load collections: ${escapeHtml(error.message)}</div>`;
    }
  },

  render(collections) {
    const container = document.getElementById("collections-list");
    if (!collections.length) {
      container.innerHTML =
        '<div class="empty-state">No databases yet. Create one, or ingest a document and Filechatter will create a matching database automatically.</div>';
      return;
    }
    container.innerHTML = collections.map((collection) => this.renderCard(collection)).join("");
    collections.forEach((collection) => {
      const card = container.querySelector(`.collection-card[data-name="${CSS.escape(collection.name)}"]`);
      if (card) card._storedSettings = this.settingSnapshot(collection);
    });
    container.querySelectorAll("[data-save]").forEach((button) => {
      button.addEventListener("click", () => this.saveCollection(button.dataset.save, true));
    });
    container.querySelectorAll("[data-delete]").forEach((button) => {
      button.addEventListener("click", () => this.openDeleteDialog(button.dataset.delete));
    });
    container.querySelectorAll("input[type='checkbox'], input[type='range']").forEach((input) => {
      input.addEventListener("change", () => this.handleSettingChange(input));
    });
    container.querySelectorAll("input[type='range']").forEach((input) => {
      const valueEl = container.querySelector(`[data-weight-value="${escapeHtml(input.dataset.weight)}:${escapeHtml(input.dataset.name)}"]`);
      if (valueEl) {
        valueEl.textContent = Number(input.value).toFixed(1);
      }
      input.addEventListener("input", () => {
        const target = container.querySelector(`[data-weight-value="${input.dataset.weight}:${input.dataset.name}"]`);
        if (target) target.textContent = Number(input.value).toFixed(1);
        this.updateDirtyState(input.closest(".collection-card"));
      });
    });
    container.querySelectorAll(".collection-card").forEach((card) => this.updateDirtyState(card));
    this.updateApplyAllState();
  },

  renderCard(collection) {
    const extensions = collection.allowed_extensions && collection.allowed_extensions.length
      ? collection.allowed_extensions.map((ext) => `<span class="ext-tag">${escapeHtml(ext)}</span>`).join("")
      : '<span class="hint">all supported types</span>';
    const maybeRead = collection.read_allowed !== false;
    const maybeWrite = collection.write_allowed === true;
    const card = `
      <div class="collection-card" data-name="${escapeHtml(collection.name)}">
        <div class="collection-controls">
          <div class="collection-control-header">
            <div class="permission-toggles">
              <label class="switch-inline"><input type="checkbox" data-role="read" data-name="${escapeHtml(collection.name)}" ${maybeRead ? "checked" : ""}> Read-enabled</label>
              <label class="switch-inline"><input type="checkbox" data-role="write" data-name="${escapeHtml(collection.name)}" ${maybeWrite ? "checked" : ""}> Write-enabled</label>
            </div>
            <button class="btn btn-primary" data-save="${escapeHtml(collection.name)}">Apply</button>
          </div>
          <div class="weight-heading">Retrieval weights <span class="hint">0 = disabled, 10 = max</span></div>
          <div class="weight-editor">
            <label class="slider-row">
              <span>Semantic</span>
              <input type="range" min="0" max="10" step="0.1" value="${Number(collection.semantic_weight ?? 1).toFixed(1)}" data-weight="semantic" data-name="${escapeHtml(collection.name)}">
              <b data-weight-value="semantic:${escapeHtml(collection.name)}">${Number(collection.semantic_weight ?? 1).toFixed(1)}</b>
            </label>
            <label class="slider-row">
              <span>Keyword</span>
              <input type="range" min="0" max="10" step="0.1" value="${Number(collection.keyword_weight ?? 2).toFixed(1)}" data-weight="keyword" data-name="${escapeHtml(collection.name)}">
              <b data-weight-value="keyword:${escapeHtml(collection.name)}">${Number(collection.keyword_weight ?? 2).toFixed(1)}</b>
            </label>
            <label class="slider-row">
              <span>Metadata</span>
              <input type="range" min="0" max="10" step="0.1" value="${Number(collection.metadata_weight ?? 4).toFixed(1)}" data-weight="metadata" data-name="${escapeHtml(collection.name)}">
              <b data-weight-value="metadata:${escapeHtml(collection.name)}">${Number(collection.metadata_weight ?? 4).toFixed(1)}</b>
            </label>
          </div>
        </div>
        <div class="collection-info">
          <div class="collection-info-header">
            <div class="collection-name">${escapeHtml(collection.name)}</div>
            <button class="btn btn-danger" data-delete="${escapeHtml(collection.name)}">Delete</button>
          </div>
          <div class="collection-desc">${escapeHtml(collection.description || "")}</div>
          <div class="collection-meta">
            <span>Documents: <b>${collection.document_count}</b></span>
            <span>Chunks: <b>${collection.chunk_count}</b></span>
            <span>File types: ${extensions}</span>
            <span>Created: <b>${formatDate(collection.created_at)}</b></span>
            <span>Last updated: <b>${formatDate(collection.last_updated)}</b></span>
          </div>
        </div>
      </div>`;
    return card;
  },

  settingSnapshot(collection) {
    return {
      read_allowed: collection.read_allowed !== false,
      write_allowed: collection.write_allowed === true,
      semantic_weight: Number(collection.semantic_weight ?? 1),
      keyword_weight: Number(collection.keyword_weight ?? 2),
      metadata_weight: Number(collection.metadata_weight ?? 4),
    };
  },

  handleSettingChange(input) {
    const card = input.closest(".collection-card");
    if (!card) return;
    if (input.dataset.role === "read" && !input.checked) {
      card.querySelectorAll('[data-weight]').forEach((slider) => {
        slider.value = "0";
        const value = card.querySelector(`[data-weight-value="${slider.dataset.weight}:${slider.dataset.name}"]`);
        if (value) value.textContent = "0.0";
      });
    } else if (input.dataset.role === "read" && input.checked && this.allWeightsZero(card)) {
      const semantic = card.querySelector('[data-weight="semantic"]');
      semantic.value = "1.0";
      const value = card.querySelector(`[data-weight-value="semantic:${semantic.dataset.name}"]`);
      if (value) value.textContent = "1.0";
    } else if (input.dataset.weight && this.allWeightsZero(card)) {
      card.querySelector('[data-role="read"]').checked = false;
    } else if (input.dataset.weight) {
      card.querySelector('[data-role="read"]').checked = true;
    }
    this.updateDirtyState(card);
  },

  allWeightsZero(card) {
    return Array.from(card.querySelectorAll('[data-weight]'))
      .every((slider) => Number(slider.value) === 0);
  },

  settingsEqual(left, right) {
    return left.read_allowed === right.read_allowed
      && left.write_allowed === right.write_allowed
      && left.semantic_weight === right.semantic_weight
      && left.keyword_weight === right.keyword_weight
      && left.metadata_weight === right.metadata_weight;
  },

  updateDirtyState(card) {
    if (!card) return;
    const dirty = !this.settingsEqual(this.collectionPayload(card), card._storedSettings);
    const button = card.querySelector("[data-save]");
    button.disabled = !dirty;
    button.classList.toggle("btn-primary", dirty);
    button.classList.toggle("btn-reset", !dirty);
    this.updateApplyAllState();
  },

  updateApplyAllState() {
    const button = document.getElementById("btn-apply-all");
    const dirty = Array.from(document.querySelectorAll(".collection-card"))
      .some((card) => !this.settingsEqual(this.collectionPayload(card), card._storedSettings));
    button.disabled = !dirty;
    button.classList.toggle("btn-primary", dirty);
    button.classList.toggle("btn-reset", !dirty);
  },

  collectionPayload(card) {
    return {
      read_allowed: card.querySelector('[data-role="read"]').checked,
      write_allowed: card.querySelector('[data-role="write"]').checked,
      semantic_weight: Number(card.querySelector('[data-weight="semantic"]').value),
      keyword_weight: Number(card.querySelector('[data-weight="keyword"]').value),
      metadata_weight: Number(card.querySelector('[data-weight="metadata"]').value),
    };
  },

  async saveCollection(name, refresh = true) {
    const card = document.querySelector(`.collection-card[data-name="${CSS.escape(name)}"]`);
    if (!card) return;

    try {
      const response = await api.updateCollection(name, this.collectionPayload(card));
      card._storedSettings = this.settingSnapshot(response.collection || this.collectionPayload(card));
      this.updateDirtyState(card);
      if (refresh) {
        showToast(`Database '${name}' updated`);
        app.refreshStatus();
      }
    } catch (error) {
      if (refresh) showToast(`Could not update '${name}': ${error.message}`);
      return false;
    }
    return true;
  },

  async saveAll() {
    const button = document.getElementById("btn-apply-all");
    const cards = Array.from(document.querySelectorAll(".collection-card"))
      .filter((card) => !this.settingsEqual(this.collectionPayload(card), card._storedSettings));
    if (!cards.length) return;
    button.disabled = true;
    try {
      const responses = await Promise.all(cards.map((card) => api.updateCollection(card.dataset.name, this.collectionPayload(card))));
      responses.forEach((response, index) => {
        cards[index]._storedSettings = this.settingSnapshot(response.collection || this.collectionPayload(cards[index]));
        this.updateDirtyState(cards[index]);
      });
      showToast("All database settings updated");
      app.refreshStatus();
    } catch (error) {
      showToast(`Could not apply all settings: ${error.message}`);
    } finally {
      this.updateApplyAllState();
    }
  },

  async openCreateDialog() {
    document.getElementById("create-name").value = "";
    document.getElementById("create-description").value = "";
    document.getElementById("create-read-enabled").checked = true;
    document.getElementById("create-write-enabled").checked = false;
    document.getElementById("create-semantic-weight").value = "1.0";
    document.getElementById("create-keyword-weight").value = "2.0";
    document.getElementById("create-metadata-weight").value = "4.0";
    document.getElementById("create-semantic-value").textContent = "1.0";
    document.getElementById("create-keyword-value").textContent = "2.0";
    document.getElementById("create-metadata-value").textContent = "4.0";
    document.getElementById("create-error").textContent = "";
    document.getElementById("ext-advanced").checked = false;
    this.selectedExts = new Set();
    // Lazy-load the catalog if it hasn't arrived yet, so the selector is never empty.
    if (!this.catalog.length) {
      try {
        this.catalog = (await api.fileTypes()).categories || [];
      } catch (_) {
        /* leave empty; user can still create with all-types default */
      }
    }
    this.renderExtSelector();
    document.getElementById("dialog-create").showModal();
  },

  availableExts(category) {
    return category.extensions.filter((e) => e.available).map((e) => e.ext);
  },

  renderExtSelector() {
    const advanced = document.getElementById("ext-advanced").checked;
    const container = document.getElementById("ext-categories");
    container.innerHTML = "";

    this.catalog.forEach((category) => {
      const available = this.availableExts(category);
      const selectedCount = available.filter((ext) => this.selectedExts.has(ext)).length;

      const block = document.createElement("div");
      block.className = "ext-category";

      // Category header with a tri-state select-all checkbox.
      const header = document.createElement("label");
      header.className = "ext-cat-head";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = selectedCount > 0 && selectedCount === available.length;
      box.indeterminate = selectedCount > 0 && selectedCount < available.length;
      box.addEventListener("change", () => {
        if (box.checked) available.forEach((ext) => this.selectedExts.add(ext));
        else available.forEach((ext) => this.selectedExts.delete(ext));
        this.renderExtSelector();
      });
      const title = document.createElement("span");
      title.innerHTML = `<b>${escapeHtml(category.category)}</b> <span class="hint">${selectedCount}/${available.length}</span>`;
      header.append(box, title);
      block.appendChild(header);

      if (advanced) {
        const grid = document.createElement("div");
        grid.className = "ext-grid";
        category.extensions.forEach((item) => {
          const label = document.createElement("label");
          label.className = "ext-item" + (item.available ? "" : " unavailable");
          const input = document.createElement("input");
          input.type = "checkbox";
          input.value = item.ext;
          input.checked = this.selectedExts.has(item.ext);
          input.disabled = !item.available;
          if (!item.available) label.title = `Needs ${item.dependency}`;
          input.addEventListener("change", () => {
            if (input.checked) this.selectedExts.add(item.ext);
            else this.selectedExts.delete(item.ext);
            this.renderExtSelector();
          });
          label.append(input, document.createTextNode(" " + item.ext));
          grid.appendChild(label);
        });
        block.appendChild(grid);
      }
      container.appendChild(block);
    });
  },

  async submitCreate(event) {
    event.preventDefault();
    const errorBox = document.getElementById("create-error");
    errorBox.textContent = "";
    const name = document.getElementById("create-name").value.trim().toLowerCase();
    const description = document.getElementById("create-description").value.trim();
    const allowed = Array.from(this.selectedExts);
    const payload = {
      name,
      description,
      allowed_extensions: allowed,
      read_allowed: document.getElementById("create-read-enabled").checked,
      write_allowed: document.getElementById("create-write-enabled").checked,
      semantic_weight: Number(document.getElementById("create-semantic-weight").value),
      keyword_weight: Number(document.getElementById("create-keyword-weight").value),
      metadata_weight: Number(document.getElementById("create-metadata-weight").value),
    };

    try {
      await api.createCollection(payload);
      document.getElementById("dialog-create").close();
      showToast(`Database '${name}' created`);
      await this.refresh();
      app.refreshStatus();
    } catch (error) {
      errorBox.textContent = error.message;
    }
  },

  openDeleteDialog(name) {
    this.pendingDelete = name;
    document.getElementById("delete-message").innerHTML =
      `Delete database <b>${escapeHtml(name)}</b> and all its indexed data? This cannot be undone.`;
    document.getElementById("delete-error").textContent = "";
    document.getElementById("dialog-delete").showModal();
  },

  async submitDelete(event) {
    event.preventDefault();
    const errorBox = document.getElementById("delete-error");
    errorBox.textContent = "";
    try {
      await api.deleteCollection(this.pendingDelete);
      document.getElementById("dialog-delete").close();
      showToast(`Database '${this.pendingDelete}' deleted`);
      this.pendingDelete = null;
      await this.refresh();
      app.refreshStatus();
    } catch (error) {
      errorBox.textContent = error.message;
    }
  },
};

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

function formatDate(isoString) {
  if (!isoString) return "–";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return isoString;
  return date.toLocaleString();
}
