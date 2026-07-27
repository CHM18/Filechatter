"""Agent loop for the built-in chat.

Drives a tool-calling conversation against an OpenAI-compatible provider,
enforcing permissions and scoping tool calls to the databases the user
selected for reading and writing.

Design notes
------------
- Reads are scoped deterministically: the agent overrides collection_name for
  content-read tools to the selected read set, regardless of what the model
  passes. The model never has to pick a read database.
- Writes are constrained by schema: with one write database the target is
  auto-filled; with several the model must choose from an enum, and if it fails
  to (or omits it) the agent asks the user which database to use.
- "Ask" tools pause the loop with a confirmation request. Confirmations are
  resolved by the caller and fed back as `decisions`.

The loop is a generator of event dicts (see EVENT TYPES) so the server can
stream them to the browser as Server-Sent Events. State between turns lives in
the `messages` list (OpenAI chat format), which the session store owns.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterator

import permissions
import runtime
import tool_registry
from llm_providers import ProviderError

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8

# Read tools whose collection_name the agent scopes to the read selection.
READ_SCOPED_TOOLS = {"search_documents", "list_sources"}


def _tool_calls_needing_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tool calls from the last assistant message that have no matching
    tool result yet (i.e. still pending resolution)."""
    last_assistant = None
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            last_assistant = message
            break
    if last_assistant is None:
        return []
    answered_ids = {
        message.get("tool_call_id")
        for message in messages
        if message.get("role") == "tool"
    }
    return [call for call in last_assistant["tool_calls"] if call["id"] not in answered_ids]


def build_chat_tools(
    read_cols: list[str], write_cols: list[str], perms: dict
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build the tool schema list for the model and a name->access plan.

    Denied tools are excluded (invisible). Read tools drop collection_name (the
    agent scopes them). Write tools are excluded when no write database is
    selected; with one they drop collection_name (auto-filled); with several
    collection_name becomes an enum of the write databases.
    """
    tools: list[dict[str, Any]] = []
    plan: dict[str, str] = {}
    for spec in permissions.visible_specs(perms):
        schema = json.loads(json.dumps(spec.params_schema))  # deep copy
        props = schema.get("properties", {})

        if spec.access == tool_registry.READ:
            props.pop("collection_name", None)
        else:  # write
            if not write_cols:
                continue  # no write target -> hide write tools entirely
            if "collection_name" in props:
                if len(write_cols) == 1:
                    props.pop("collection_name")
                else:
                    props["collection_name"] = {
                        "type": "string",
                        "enum": list(write_cols),
                        "description": "Target database for this write operation.",
                    }

        schema["properties"] = props
        schema["required"] = [name for name in schema.get("required", []) if name in props]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": schema,
                },
            }
        )
        plan[spec.name] = spec.access
    return tools, plan


def _system_prompt(read_cols: list[str], write_cols: list[str]) -> str:
    manager = runtime.collections()

    def describe(names: list[str]) -> str:
        lines = []
        for name in names:
            entry = manager.get_entry(name)
            if entry is None:
                continue
            extensions = ", ".join(entry.allowed_extensions) or "any supported type"
            description = entry.description or "(no description)"
            lines.append(f"- {name}: {description} [accepts: {extensions}]")
        return "\n".join(lines) if lines else "(none)"

    return (
        "You are Filechatter, a retrieval-augmented assistant over the user's local "
        "document databases. Use search_documents to ground answers in the user's files "
        "before responding, and cite the sources you used. Only call write tools when the "
        "user clearly asks to ingest or delete content.\n\n"
        f"Databases available for reading:\n{describe(read_cols)}\n\n"
        f"Databases available for writing:\n{describe(write_cols)}\n\n"
        "When a write is requested and several write databases are available, choose the one "
        "whose description and accepted file types best match the content; if none clearly "
        "fits, ask the user."
    )


def _resolve_write_collection(
    raw_arguments: dict[str, Any], write_cols: list[str], decision: dict[str, Any] | None
) -> tuple[str | None, bool]:
    """Pick the target write database. Returns (collection, needs_choice).

    needs_choice is True when several write databases are available and neither
    the user's decision nor the model proposed a valid one."""
    if decision and decision.get("collection") in write_cols:
        return decision["collection"], False
    if len(write_cols) == 1:
        return write_cols[0], False
    proposed = raw_arguments.get("collection_name")
    if proposed in write_cols:
        return proposed, False
    if not write_cols:
        return None, False
    return None, True


def _summarize_result(result: Any) -> Any:
    """Trim tool results before sending them back to the model / browser."""
    if isinstance(result, list):
        return result[:10]
    return result


def run(
    provider: Any,
    messages: list[dict[str, Any]],
    read_cols: list[str],
    write_cols: list[str],
    decisions: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Run/resume the agent loop, yielding event dicts.

    EVENT TYPES
      {"type": "token", "text": str}
      {"type": "tool_call", "id","name","arguments"}
      {"type": "tool_result", "id","name","ok":bool,"result":Any}
      {"type": "need_confirmation", "pending":[{id,name,arguments,kind,options?}]}
      {"type": "done", "content": str}
      {"type": "error", "message": str}
    """
    decisions = decisions or {}
    perms = runtime.settings().get()["permissions"]
    tools, plan = build_chat_tools(read_cols, write_cols, perms)

    # Ensure a system prompt leads the conversation.
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": _system_prompt(read_cols, write_cols)})

    try:
        for _iteration in range(MAX_ITERATIONS):
            pending = _tool_calls_needing_results(messages)

            if pending:
                confirmations: list[dict[str, Any]] = []
                for call in pending:
                    name = call["function"]["name"]
                    try:
                        raw_args = json.loads(call["function"].get("arguments") or "{}")
                    except json.JSONDecodeError:
                        raw_args = {}
                    access = plan.get(name, tool_registry.READ)
                    decision = decisions.get(call["id"])
                    resolved = permissions.resolve(name, perms)

                    # Denied by user during confirmation.
                    if decision is not None and decision.get("approved") is False:
                        messages.append(_tool_message(call["id"], {"status": "denied by user"}))
                        yield {"type": "tool_result", "id": call["id"], "name": name,
                               "ok": False, "result": "Denied by user."}
                        continue

                    approved = bool(decision and decision.get("approved"))
                    needs_approval = resolved == permissions.ASK and not approved

                    write_collection = None
                    needs_choice = False
                    if access == tool_registry.WRITE:
                        write_collection, needs_choice = _resolve_write_collection(
                            raw_args, write_cols, decision
                        )

                    # One combined confirmation: approval and/or database choice.
                    if needs_approval or needs_choice:
                        confirmations.append({
                            "id": call["id"],
                            "name": name,
                            "arguments": raw_args,
                            "needs_approval": needs_approval,
                            "options": list(write_cols) if needs_choice else None,
                        })
                        continue

                    scoped_args = dict(raw_args)
                    if access == tool_registry.READ:
                        if name in READ_SCOPED_TOOLS:
                            scoped_args["collection_name"] = list(read_cols)
                    elif write_collection is not None:
                        scoped_args["collection_name"] = write_collection

                    yield {"type": "tool_call", "id": call["id"], "name": name, "arguments": scoped_args}
                    try:
                        permissions.enforce(name)  # deny safety net
                        result = tool_registry.execute(name, scoped_args)
                        messages.append(_tool_message(call["id"], _summarize_result(result)))
                        yield {"type": "tool_result", "id": call["id"], "name": name,
                               "ok": True, "result": _summarize_result(result)}
                    except Exception as exc:  # tool errors are fed back to the model
                        messages.append(_tool_message(call["id"], {"error": str(exc)}))
                        yield {"type": "tool_result", "id": call["id"], "name": name,
                               "ok": False, "result": str(exc)}

                if confirmations:
                    yield {"type": "need_confirmation", "pending": confirmations}
                    return
                continue  # all resolved -> ask the model again

            # No pending tool calls: get the next assistant turn.
            assistant_message: dict[str, Any] = {"role": "assistant", "content": ""}
            for kind, value in provider.stream(messages, tools or None):
                if kind == "token":
                    yield {"type": "token", "text": value}
                elif kind == "message":
                    assistant_message = value
            messages.append(assistant_message)

            if assistant_message.get("tool_calls"):
                continue  # resolved on the next iteration
            yield {"type": "done", "content": assistant_message.get("content", "")}
            return

        yield {"type": "error", "message": "Reached the tool-call limit without a final answer."}

    except ProviderError as exc:
        yield {"type": "error", "message": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Chat agent crashed")
        yield {"type": "error", "message": f"Internal error: {exc}"}


def _tool_message(tool_call_id: str, content: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(content, ensure_ascii=False, default=str),
    }
