"""Permission resolution and enforcement for Filechatter tools.

The user controls whether the LLM may call tools, split into read-only and
write operations, each set to "ask", "allow", or "deny". An optional advanced
mode overrides the decision per individual tool.

Resolution order (see settings_manager for the stored shape):
  1. advanced mode + an explicit per-tool entry  -> that value wins
  2. otherwise                                    -> the group default for the
                                                     tool's access class (read/write)

"deny" means the tool is invisible to the model (never listed) and rejected
if called anyway. "ask" requires confirmation before running; how that is
handled depends on the caller:
  - web chat (M4): the agent loop pauses and asks the user in the browser
  - MCP hosts:      governed by mcp_hosts.on_ask (host_confirm | elicit | deny)

Enforcement here is caller-agnostic and authoritative: even a stale MCP proxy
that still advertises a now-denied tool is blocked at execution time.
"""
from __future__ import annotations

import runtime
import tool_registry

ALLOW = "allow"
ASK = "ask"
DENY = "deny"

READ_TOOLS = "read_tools"
INGEST_TOOLS = "ingest_tools"
COLLECTION_TOOLS = "collection_tools"

TOOL_GROUPS = {
    "search_documents": READ_TOOLS,
    "list_sources": READ_TOOLS,
    "get_document_support": READ_TOOLS,
    "ingest_file": INGEST_TOOLS,
    "start_ingest_directory": INGEST_TOOLS,
    "get_ingest_status": INGEST_TOOLS,
    "get_ingest_log": INGEST_TOOLS,
    "create_collection": COLLECTION_TOOLS,
    "delete_collection": COLLECTION_TOOLS,
    "clear_index": COLLECTION_TOOLS,
}

BUCKET_LABELS = {
    READ_TOOLS: "Read tools",
    INGEST_TOOLS: "Ingest tools",
    COLLECTION_TOOLS: "Collection administration",
}

# Fallback group defaults if settings somehow omit a group.
_DEFAULT_GROUP = {
    READ_TOOLS: ALLOW,
    INGEST_TOOLS: ASK,
    COLLECTION_TOOLS: ASK,
}


def tool_group(tool_name: str) -> str:
    """Return the global permission bucket for a registered tool."""
    return TOOL_GROUPS.get(tool_name, INGEST_TOOLS)


def _group_value(perms: dict, group: str) -> str:
    groups = perms.get("groups", {})
    if group in groups:
        return groups[group]
    # Migrate the old two-bucket settings shape without rewriting the file.
    if group == READ_TOOLS and "read" in groups:
        return groups["read"]
    if group in {INGEST_TOOLS, COLLECTION_TOOLS} and "write" in groups:
        return groups["write"]
    return _DEFAULT_GROUP[group]

# Caller contexts.
CLIENT_API = "api"   # direct REST / web chat (M4 handles ask itself)
CLIENT_MCP = "mcp"   # an MCP host via the stdio proxy


class PermissionDenied(RuntimeError):
    """Raised when a tool call is blocked by the permission policy."""


def _permissions() -> dict:
    return runtime.settings().get()["permissions"]


def resolve(tool_name: str, permissions: dict | None = None) -> str:
    """Return the effective decision (allow/ask/deny) for a tool."""
    spec = tool_registry.get_spec(tool_name)  # raises KeyError for unknown tools
    perms = permissions if permissions is not None else _permissions()
    mode = perms.get("mode", "simple")
    if mode == "advanced":
        override = perms.get("tools", {}).get(tool_name)
        if override in (ALLOW, ASK, DENY):
            return override
    return _group_value(perms, tool_group(tool_name))


def is_visible(tool_name: str, permissions: dict | None = None) -> bool:
    """A tool is visible to the model unless it resolves to deny."""
    return resolve(tool_name, permissions) != DENY


def visible_specs(permissions: dict | None = None) -> list[tool_registry.ToolSpec]:
    perms = permissions if permissions is not None else _permissions()
    return [spec for spec in tool_registry.list_specs() if resolve(spec.name, perms) != DENY]


def visible_tool_names(permissions: dict | None = None) -> list[str]:
    return [spec.name for spec in visible_specs(permissions)]


def describe(tool_name: str, permissions: dict | None = None) -> dict:
    """Full decision for a tool, for API responses and the settings UI."""
    spec = tool_registry.get_spec(tool_name)
    perms = permissions if permissions is not None else _permissions()
    override = None
    if perms.get("mode") == "advanced":
        override = perms.get("tools", {}).get(tool_name)
        if override not in (ALLOW, ASK, DENY):
            override = None
    return {
        "name": tool_name,
        "access": spec.access,
        "permission_group": tool_group(tool_name),
        "permission": resolve(tool_name, perms),
        "group_default": _group_value(perms, tool_group(tool_name)),
        "has_override": override is not None,
    }


def enforce(tool_name: str, client: str = CLIENT_API) -> str:
    """Check whether a tool may execute for the given caller.

    Returns the resolved decision on success; raises PermissionDenied when the
    call must be blocked. Callers that resolve to "ask" are allowed to proceed
    here - the confirmation itself happens above this layer (browser dialog for
    web chat; the host's own prompt for MCP under host_confirm).
    """
    decision = resolve(tool_name)
    if decision == DENY:
        raise PermissionDenied(
            f"Tool '{tool_name}' is disabled by the permission settings (deny)."
        )
    if decision == ASK and client == CLIENT_MCP:
        on_ask = runtime.settings().get().get("mcp_hosts", {}).get("on_ask", "host_confirm")
        if on_ask == DENY:
            raise PermissionDenied(
                f"Tool '{tool_name}' requires confirmation, but confirmation from MCP "
                f"hosts is disabled (mcp_hosts.on_ask = 'deny')."
            )
        # host_confirm / elicit: the host handles the confirmation prompt; the
        # server allows the call to proceed.
    return decision
