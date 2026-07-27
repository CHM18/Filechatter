// Permissions panel: group controls + advanced per-tool overrides, bucketed
// by Read / Write with a tri-state "mixed" indicator on group buttons.
"use strict";

const PERM_VALUES = ["ask", "allow", "deny"];
const PERM_LABELS = { ask: "Ask", allow: "Allow", deny: "Deny" };
const GROUPS = [
  { key: "read", label: "Read operations", hint: "search, list, status, logs" },
  { key: "write", label: "Write operations", hint: "ingest, clear" },
];

const permissionsPanel = {
  settings: null,
  tools: [],
  loaded: false,

  async init() {
    document.getElementById("advanced-toggle").addEventListener("change", (event) => {
      this.permissions().mode = event.target.checked ? "advanced" : "simple";
      this.commit();
    });
    document.getElementById("on-ask-select").addEventListener("change", (event) => {
      this.saveOnAsk(event.target.value);
    });
  },

  permissions() {
    return this.settings.permissions;
  },

  async refresh() {
    try {
      const [settings, toolsPayload] = await Promise.all([api.settings(), api.listTools()]);
      this.settings = settings;
      this.tools = toolsPayload.tools || [];
      this.loaded = true;
      this.render();
    } catch (error) {
      showToast(`Failed to load permissions: ${error.message}`);
    }
  },

  toolsInGroup(groupKey) {
    return this.tools
      .filter((tool) => tool.access === groupKey)
      .sort((a, b) => a.name.localeCompare(b.name));
  },

  // A group is "mixed" when any of its tools has a stored override that differs
  // from the group value - true regardless of mode, so simple mode still warns
  // about overrides stashed while in advanced mode.
  isMixed(groupKey) {
    const perms = this.permissions();
    const groupValue = perms.groups[groupKey];
    return this.toolsInGroup(groupKey).some((tool) => {
      const override = perms.tools[tool.name];
      return override !== undefined && override !== groupValue;
    });
  },

  // Resolved value shown on a per-tool selector.
  toolValue(tool) {
    const perms = this.permissions();
    const override = perms.tools[tool.name];
    return override !== undefined ? override : perms.groups[tool.access];
  },

  render() {
    if (!this.loaded) return;
    const perms = this.permissions();
    const advanced = perms.mode === "advanced";

    document.getElementById("advanced-toggle").checked = advanced;
    document.getElementById("advanced-extras").classList.toggle("hidden", !advanced);
    document.getElementById("on-ask-select").value = this.settings.mcp_hosts.on_ask;

    const container = document.getElementById("perm-container");
    container.innerHTML = "";

    if (!advanced) {
      const card = el("div", "card wide");
      GROUPS.forEach((group) => {
        const row = el("div", "perm-row");
        row.appendChild(this.nameCell(group.label, group.hint));
        row.appendChild(
          this.selector(perms.groups[group.key], this.isMixed(group.key), (value) =>
            this.setGroup(group.key, value)
          )
        );
        card.appendChild(row);
      });
      container.appendChild(card);
      return;
    }

    // Advanced: one bucket per group, group buttons as the bucket header.
    GROUPS.forEach((group) => {
      const bucket = el("div", "perm-bucket card wide");

      const head = el("div", "perm-bucket-head");
      head.appendChild(this.nameCell(group.label, "applies to every tool below"));
      head.appendChild(
        this.selector(perms.groups[group.key], this.isMixed(group.key), (value) =>
          this.setGroup(group.key, value)
        )
      );
      bucket.appendChild(head);

      this.toolsInGroup(group.key).forEach((tool) => {
        const row = el("div", "perm-row perm-tool-row");
        const overridden = perms.tools[tool.name] !== undefined
          && perms.tools[tool.name] !== perms.groups[group.key];
        row.appendChild(this.nameCell(tool.name, overridden ? "overridden" : "follows group", true));
        row.appendChild(
          this.selector(this.toolValue(tool), false, (value) =>
            this.setTool(tool, value)
          )
        );
        bucket.appendChild(row);
      });

      container.appendChild(bucket);
    });
  },

  nameCell(label, hint, mono = false) {
    const cell = el("div", "perm-name");
    const name = el("span", mono ? "perm-label mono" : "perm-label");
    name.textContent = label;
    const sub = el("span", "hint");
    sub.textContent = hint;
    cell.append(name, sub);
    return cell;
  },

  // Segmented Ask/Allow/Deny control. `mixed` fades the selected button.
  selector(current, mixed, onChange) {
    const seg = el("div", "seg");
    PERM_VALUES.forEach((value) => {
      const button = el("button", "seg-btn");
      button.type = "button";
      button.textContent = PERM_LABELS[value];
      if (value === current) {
        button.classList.add("sel");
        if (mixed) {
          button.classList.add("mixed");
          button.title = "Some tools in this group are overridden. Click to reset them.";
        }
      }
      button.addEventListener("click", () => onChange(value));
      seg.appendChild(button);
    });
    return seg;
  },

  // --- mutations (operate on local state, then commit the whole block) ---

  setGroup(groupKey, value) {
    const perms = this.permissions();
    perms.groups[groupKey] = value;
    // Reset: clear every override in this group so it becomes uniform/opaque.
    this.toolsInGroup(groupKey).forEach((tool) => {
      delete perms.tools[tool.name];
    });
    this.commit();
  },

  setTool(tool, value) {
    const perms = this.permissions();
    perms.mode = "advanced";
    if (value === perms.groups[tool.access]) {
      delete perms.tools[tool.name]; // matches group -> no override
    } else {
      perms.tools[tool.name] = value;
    }
    this.commit();
  },

  async commit() {
    try {
      this.settings = await api.setPermissions(this.permissions());
      const toolsPayload = await api.listTools();
      this.tools = toolsPayload.tools || [];
      this.render();
      app.refreshStatus();
    } catch (error) {
      showToast(`Could not save: ${error.message}`);
      this.refresh();
    }
  },

  async saveOnAsk(value) {
    try {
      this.settings = await api.updateSettings({ mcp_hosts: { on_ask: value } });
    } catch (error) {
      showToast(`Could not save: ${error.message}`);
      this.refresh();
    }
  },
};

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}
