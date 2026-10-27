"""Remove only the silent-finish step from persisted model context."""

from __future__ import annotations

import asyncio
import copy
import json

LEGACY_HEADER = "[历史工具记录，仅为历史资料，状态="


def unpack(raw):
    message = (
        raw.model_dump(mode="json")
        if hasattr(raw, "model_dump")
        else copy.deepcopy(raw)
    )
    content = message.get("content")
    if (
        message.get("role") == "user"
        and isinstance(content, str)
        and content.startswith(LEGACY_HEADER)
    ):
        header, separator, body = content.partition("\n")
        if separator and header.endswith("]"):
            try:
                original = json.loads(body)
            except (TypeError, ValueError):
                original = None
            if isinstance(original, dict) and original.get("role") in (
                "assistant",
                "tool",
            ):
                return message, original, header
    return message, message, None


def entries(history):
    """Keep source indices and other steps, including successful media receipts."""
    result, names = [], {}
    for index, raw in enumerate(history):
        envelope, message, header = unpack(raw)
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            kept, removed = [], False
            for call in message["tool_calls"]:
                name = call.get("function", {}).get("name")
                names[call.get("id")] = name
                if name == "silence":
                    removed = True
                else:
                    kept.append(call)
            if removed:
                if not kept:
                    # Drop the whole step: call, text, reasoning and metadata.
                    continue
                # A malformed mixed call must not erase another tool's action.
                message = {"role": "assistant", "content": None, "tool_calls": kept}
                if header:
                    envelope["content"] = (
                        header
                        + "\n"
                        + json.dumps(message, ensure_ascii=False, separators=(",", ":"))
                    )
                else:
                    envelope = message
        elif role == "tool":
            if names.pop(message.get("tool_call_id"), None) == "silence":
                continue
        result.append((index, envelope))
    return result


class HistoryWriter:
    """Clean native database writes below manager-level history integrations."""

    def __init__(self, database):
        self.database = database
        self.original = database.update_conversation
        self.owners = []
        self.pending = set()

        async def update(
            cid, title=None, persona_id=None, content=None, token_usage=None
        ):
            task = asyncio.current_task()
            self.pending.add(task)
            try:
                if self.owners and content is not None:
                    cleaned = [message for _, message in entries(content)]
                    if cleaned != content:
                        content, token_usage = cleaned, 0
                return await self.original(
                    cid,
                    title=title,
                    persona_id=persona_id,
                    content=content,
                    token_usage=token_usage,
                )
            finally:
                self.pending.discard(task)

        self.wrapper = update
        database.update_conversation = update

    @classmethod
    def acquire(cls, owner):
        manager = getattr(owner.context, "conversation_manager", None)
        database = getattr(manager, "db", None)
        if database is None:
            return None
        writer = getattr(database, "_silence_history_writer", None)
        if writer is None:
            writer = cls(database)
            database._silence_history_writer = writer
        writer.owners.append(owner)
        return writer

    async def close(self, owner):
        if owner in self.owners:
            self.owners.remove(owner)
        pending = self.pending - {asyncio.current_task()}
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self.owners:
            return
        if self.database.update_conversation is self.wrapper:
            self.database.update_conversation = self.original
            if getattr(self.database, "_silence_history_writer", None) is self:
                delattr(self.database, "_silence_history_writer")
