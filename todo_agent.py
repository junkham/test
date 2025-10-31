from __future__ import annotations
import re
from typing import List
import logging

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.agent import Agent
from openhands.core.config import LLMConfig
from openhands.events.action import MessageAction
from openhands.llm.llm import LLM

logger = logging.getLogger(__name__)

class TodoCodeActAgent(Agent):
    """分层 agent：先规划 todo，再逐步完成"""

    def __init__(self, llm_config: LLMConfig, config=None):
        super().__init__(llm_config, config)
        self.llm = LLM(llm_config)
        self.codeact = CodeActAgent(llm_config, config)
        self.todo: List[str] = []

    def _generate_todo(self, task: str) -> List[str]:
        prompt = (
            "You are a planner. Break down the following task into concise, "
            "ordered steps (one per line, verb-starting, no numbering):\n\n"
            f"{task}\n\n"
            "Steps:"
        )
        try:
            resp = self.llm.completion(prompt).text
            steps = [line.strip('- ') for line in resp.splitlines() if 1 <= len(line.strip()) <= 100]
            return steps[:10]  # limit
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}")
            return ["Analyze the task", "Plan next steps", "Execute", "Verify"]

    def step(self, state) -> MessageAction:
        # Step 1: 初始化 TODO（仅一次）
        if not self.todo:
            initial_msg = ""
            for evt in state.history:
                if isinstance(evt, MessageAction) and evt.source == 'user':
                    initial_msg = evt.content
                    break
            self.todo = self._generate_todo(initial_msg)
            logger.info(f"[TodoCodeAct] Generated {len(self.todo)} steps: {self.todo}")

        # Step 2: 构造带进度提示的系统 message
        if self.todo:
            current_step = self.todo[0]
            prefix_msg = (
                f"[Planner] Current task: {current_step}\n"
                f"Remaining: {len(self.todo) - 1}\n"
                "Please work on this step. If done, say 'STEP COMPLETED'."
            )
        else:
            return MessageAction(content="All steps completed.", source='agent')

        # 构造新的 history：原 history + 新 prefix（作为 system 或 user）
        extended_history = [
            MessageAction(content=prefix_msg, source='user')
        ] + state.history

        # Step 3: 使用 CodeActAgent 执行（注意：这里仍存在 history 篡改风险）
        action = self.codeact.step(state.copy(update={"history": extended_history}))

        # Step 4: 如果 agent 完成当前 step，则更新 todo
        if isinstance(action, MessageAction):
            if 'STEP COMPLETED' in action.content.upper():
                self.todo.pop(0)
                logger.info(f"[TodoCodeAct] Step completed. {len(self.todo)} remain.")
                # 返回新指令或结束
                if self.todo:
                    return MessageAction(
                        content=f"继续：{self.todo[0]}",
                        keep_details=True
                    )
                else:
                    return MessageAction(content="Task completed.", source='agent')

        return action
