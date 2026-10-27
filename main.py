"""Expose a general silent-finish tool using AstrBot's native tool lifecycle."""

from astrbot.api import llm_tool
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Star

from .history import HistoryWriter, entries


class Silence(Star):
    """End only the current agent turn without sending another message."""

    def __init__(self, context):
        super().__init__(context)
        self._history_writer = HistoryWriter.acquire(self)

    def lcm_context_rules(self):
        return {
            "owner": "astrbot_plugin_silence",
            "history_tools": {"silence": "turn"},
            "history_projector": entries,
        }

    @llm_tool(name="silence")
    async def silence(self, event: AstrMessageEvent) -> None:
        """选择沉默，立即结束当前轮，不再生成或发送后续回复。

        你可以直接选择沉默，也可以在发图、修改记忆或任意其他工具操作后调用。
        先根据其他工具的实际结果判断下一步，再决定是否沉默；沉默本身不表示
        其他操作已成功。将本工具作为最后一次、单独的工具调用，该次正文留空，
        不要额外说“我不回复了”或汇报沉默。已经执行的操作、已经发送的内容
        会保留。下一条消息仍按正常流程处理。
        """
        event.set_extra("silence_completed", True)
        return None

    async def terminate(self):
        if self._history_writer is not None:
            await self._history_writer.close(self)
