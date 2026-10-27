"""Exercise silent completion through AstrBot's native runner and executor."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from astrbot.api.message_components import Plain
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import ToolSet
from astrbot.core.astr_agent_run_util import run_agent
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import llm_tools

from astrbot_plugin_silence.main import Silence


class SilenceTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, with_action):
        message = AstrBotMessage()
        message.type = MessageType.GROUP_MESSAGE
        message.self_id = 'bot'
        message.session_id = message.group_id = 'group'
        message.message_id = 'silence-test'
        message.sender = MessageMember(user_id='member', nickname='Member')
        message.message = [Plain('可以不回复')]
        message.message_str = '可以不回复'
        event = AstrMessageEvent(
            message.message_str, message,
            PlatformMetadata('qq_official', 'qq_official', 'test'), 'group'
        )
        event.send = AsyncMock()
        database_write = AsyncMock()
        database = SimpleNamespace(update_conversation=database_write)

        async def save(unified_msg_origin, conversation_id, history, **kwargs):
            await database.update_conversation(conversation_id, content=history)

        manager = SimpleNamespace(db=database, update_conversation=AsyncMock(side_effect=save))
        plugin = Silence(SimpleNamespace(conversation_manager=manager))
        self.addAsyncCleanup(plugin.terminate)
        silence = copy.copy(llm_tools.get_func('silence'))
        self.assertEqual(silence.parameters.get('properties'), {})
        silence.handler = plugin.silence
        tools = [silence]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'action.json'

            async def record_action(event):
                path.write_text('{"completed": true}')
                return '{"ok": true}'

            if with_action:
                action = copy.copy(silence)
                action.name = 'record_action'
                action.handler = record_action
                tools.append(action)
            responses = []
            if with_action:
                responses.append(LLMResponse(
                    role='assistant', tools_call_name=['record_action'],
                    tools_call_args=[{}], tools_call_ids=['action-1']
                ))
            responses.append(LLMResponse(
                role='assistant', tools_call_name=['silence'],
                tools_call_args=[{}], tools_call_ids=['silence-1'],
                reasoning_content='PRIVATE_SILENCE_REASONING'
            ))
            provider = MagicMock(spec=Provider)
            provider.provider_config = {'modalities': ['text']}
            provider.meta.return_value = SimpleNamespace(
                id='test', model='test', type='openai_chat_completion'
            )
            provider.text_chat = AsyncMock(side_effect=responses)
            request = ProviderRequest(
                prompt=message.message_str, func_tool=ToolSet(tools),
                conversation=SimpleNamespace(cid='conversation', token_usage=0)
            )
            event.set_extra('provider_request', request)
            runner = ToolLoopAgentRunner()
            await runner.reset(
                provider=provider, request=request,
                run_context=ContextWrapper(context=SimpleNamespace(event=event)),
                tool_executor=FunctionToolExecutor(), agent_hooks=BaseAgentRunHooks()
            )
            replies = [reply async for reply in run_agent(
                runner, max_step=5, show_tool_use=False
            )]
            self.assertTrue(runner.done())
            self.assertIsNone(runner.get_final_llm_resp())
            self.assertEqual(replies, [])
            self.assertFalse(event.is_stopped())
            event.send.assert_not_awaited()
            self.assertEqual(provider.text_chat.await_count, 2 if with_action else 1)
            if with_action:
                self.assertEqual(json.loads(path.read_text()), {'completed': True})
            stage = SimpleNamespace(conv_manager=manager)
            await InternalAgentSubStage._save_to_history(
                stage, event, request, runner.get_final_llm_resp(),
                runner.run_context.messages, runner.stats
            )
            stage.conv_manager.update_conversation.assert_awaited_once()
            history = database_write.await_args.kwargs['content']
            results = {m['tool_call_id']: m for m in history if m.get('role') == 'tool'}
            self.assertNotIn('silence-1', results)
            self.assertNotIn('PRIVATE_SILENCE_REASONING', str(history))
            self.assertNotIn('silence-1', str(history))
            self.assertTrue(event.get_extra('silence_completed'))
            if with_action:
                self.assertTrue(json.loads(results['action-1']['content'])['ok'])
            self.assertFalse(any(
                m.get('role') == 'assistant' and not m.get('tool_calls') for m in history
            ))

    async def test_direct_silence_finishes_and_saves_history(self):
        await self._run(False)

    async def test_action_is_preserved_and_no_reply_follows_silence(self):
        await self._run(True)
