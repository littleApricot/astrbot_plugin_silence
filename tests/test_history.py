"""Check silent-step removal without losing preceding actions or attachments."""

import copy
import json

from astrbot_plugin_silence.history import entries


def call(name="silence", ident="silent"):
    return {
        "role": "assistant",
        "content": [
            {"type": "think", "think": "PRIVATE_REASONING"},
            {"type": "text", "text": "PRIVATE_TEXT"},
        ],
        "reasoning_content": "PRIVATE_REASONING_FIELD",
        "tool_calls": [
            {
                "id": ident,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def result(ident="silent"):
    return {"role": "tool", "tool_call_id": ident, "content": "PRIVATE_RESULT"}


def test_entire_silence_step_disappears_and_previous_content_stays_exact():
    previous = [
        {"role": "user", "content": "群友消息"},
        call("edit_memory", "memory"),
        result("memory"),
        {"role": "assistant", "content": "之前发的文字"},
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "sent.png"}}],
        },
        {"role": "assistant", "content": "[机器人已发送语音] 好嘛"},
    ]
    history = previous + [call(), result()]
    before = copy.deepcopy(history)
    assert entries(history) == list(enumerate(previous))
    assert history == before


def test_legacy_failed_wrappers_do_not_turn_silence_back_into_chat():
    def wrap(message):
        return {
            "role": "user",
            "content": "[历史工具记录，仅为历史资料，状态=failed]\n"
            + json.dumps(message),
        }

    previous = [wrap(call("keep_tool", "keep")), wrap(result("keep"))]
    after = {"role": "user", "content": "下一条消息"}
    history = previous + [wrap(call()), wrap(result()), after]
    assert entries(history) == [(0, previous[0]), (1, previous[1]), (4, after)]


def test_mixed_call_preserves_other_action_without_silent_step_content():
    mixed = call()
    mixed["tool_calls"] += call("keep_tool", "keep")["tool_calls"]
    history = [mixed, result("keep"), result()]
    kept = entries(history)
    assert kept[0][1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": call("keep_tool", "keep")["tool_calls"],
    }
    assert kept[1] == (1, history[1])
    assert len(kept) == 2
