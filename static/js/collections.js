// Databases panel: list, create, delete collections.
"use strict";

const collectionsPanel = {
  catalog: [], // [{category, extensions:[{ext,available,dependency}]}]
  selectedExts: new Set(),
  pendingDelete: null,

  async init() {
    document.getElementById("btn-new-collection").addEventListener("click", () => this.openCreateDialog());
    document.getElementById("btn-create-cancel").addEventListener("click", () => {
      document.getElementById("dialog-create").close();
    });
    document.getElementById("btn-delete-cancel").addEventListener("click", () => {
      document.getElementById("dialog-delete").close();
    });
    document.getElementById("form-create").addEventListener("submit", (event) => this.submitCreate(event));
    document.getElementById("form-delete").addEventListener("submit", (event) => this.submitDelete(event));
    document.getElementById("ext-advanced").addEventListener("change", () => this.renderExtSelector());

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
    container.querySelectorAll("[data-delete]").forEach((button) => {
      button.addEventListener("click", () => this.openDeleteDialog(button.dataset.delete));
    });
  },

  renderCard(collection) {
    const extensions = collection.allowed_extensions && collection.allowed_extensions.length
      ? collection.allowed_extensions.map((ext) => `<span class="ext-tag">${escapeHtml(ext)}</span>`).join("")
      : '<span class="hint">all supported types</span>';
    return `
      <div class="collection-card">
        <div class="collection-name">${escapeHtml(collection.name)}</div>
        <div class="collection-actions">
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
      </div>`;
  },

  async openCreateDialog() {
    document.getElementById("create-name").value = "";
    document.getElementById("create-description").value = "";
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

    try {
      await api.createCollection({ name, description, allowed_extensions: allowed });
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
