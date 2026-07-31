// Local long-term memory management.
"use strict";

const memoryPanel = {
  memories: [],

  init() {
    document.getElementById("memory-clear").addEventListener("click", () => this.clear());
  },

  async refresh() {
    try {
      this.memories = (await api.listMemories()).memories || [];
      this.render();
    } catch (error) {
      showToast(`Could not load memory: ${error.message}`);
    }
  },

  render() {
    const list = document.getElementById("memory-list");
    if (!this.memories.length) {
      list.innerHTML = '<div class="empty-state">No saved memory yet.</div>';
      return;
    }
    list.innerHTML = "";
    this.memories.forEach((memory) => list.appendChild(this.buildRow(memory)));
  },

  buildRow(memory) {
    const row = document.createElement("article");
    row.className = "memory-row";
    if (!memory.enabled) row.classList.add("disabled");

    const fields = document.createElement("div");
    fields.className = "memory-fields";
    const category = document.createElement("select");
    category.innerHTML = '<option value="preference">Preference</option><option value="workflow">Workflow</option>';
    category.value = memory.category;
    category.setAttribute("aria-label", "Memory category");
    const fact = document.createElement("textarea");
    fact.value = memory.fact;
    fact.rows = 2;
    fact.maxLength = 320;
    fact.setAttribute("aria-label", "Memory fact");
    fields.append(category, fact);

    const controls = document.createElement("div");
    controls.className = "memory-controls";
    const enabled = document.createElement("input");
    enabled.type = "checkbox";
    enabled.checked = memory.enabled;
    enabled.title = "Use this memory in chat";
    enabled.setAttribute("aria-label", "Use this memory in chat");
    const save = document.createElement("button");
    save.type = "button";
    save.className = "btn btn-small";
    save.textContent = "Save";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn btn-small";
    remove.textContent = "Delete";
    controls.append(enabled, save, remove);
    row.append(fields, controls);

    save.addEventListener("click", async () => {
      try {
        await api.updateMemory(memory.id, {
          category: category.value,
          fact: fact.value.trim(),
          enabled: enabled.checked,
        });
        await this.refresh();
      } catch (error) {
        showToast(`Could not save memory: ${error.message}`);
      }
    });
    enabled.addEventListener("change", () => save.click());
    remove.addEventListener("click", async () => {
      if (!window.confirm("Delete this saved memory?")) return;
      try {
        await api.deleteMemory(memory.id);
        await this.refresh();
      } catch (error) {
        showToast(`Could not delete memory: ${error.message}`);
      }
    });
    return row;
  },

  async clear() {
    if (!this.memories.length || !window.confirm("Clear all saved memory?")) return;
    try {
      await api.clearMemories();
      await this.refresh();
    } catch (error) {
      showToast(`Could not clear memory: ${error.message}`);
    }
  },
};