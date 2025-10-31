from __future__ import annotations
import re
from typing import List
import logging

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.agent import Agent
from openhands.core.config import LLMConfig
from openhands.events.action import MessageAction
from openhands.llm.llm import LLM
from openhands.core.config import AgentConfig
from openhands.core.metrics import Metrics
from openhands.events.observation import Observation

logger = logging.getLogger(__name__)


class TodoCodeActAgent(Agent):
    """A hierarchical agent that:
    1. First breaks down the user task into a TODO list
    2. Then executes each step using CodeAct
    3. Updates the plan dynamically
    """

    def __init__(self, llm_config: LLMConfig, config: AgentConfig | None = None):
        # 显式初始化 LLM 实例，避免父类未定义问题
        self.llm = LLM(llm_config)
        super().__init__(llm=self.llm, config=config or AgentConfig())

        # 初始化 CodeActAgent（使用相同的 LLM 和配置）
        self.codeact = CodeActAgent(llm=self.llm, config=config)

        # 当前待办列表
        self.todo: List[str] = []

        # 任务是否已初始化规划
        self._planning_done = False

    def reset(self):
        """Reset the agent's internal state for a new task."""
        super().reset()  # 调用父类 reset（处理 metrics 等）
        self.todo = []
        self._planning_done = False
        self.codeact.reset()  # 可选：重置底层 agent
        logger.info("[TodoCodeActAgent] Reset completed.")

    def _generate_todo(self, task: str) -> List[str]:
        """Generate a list of steps using LLM."""
        prompt = (
            "You are a helpful assistant acting as a planner. "
            "Break down the following task into clear, ordered, and executable steps.\n"
            "Each step should be concise, start with a verb, and be actionable in a Linux environment.\n"
            "Return only the steps, one per line, without numbering or bullet symbols.\n\n"
            f"Task: {task}\n\n"
            "Steps:"
        )

        try:
            response = self.llm.completion(
                messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )
            content = response.choices[0].message.content.strip()
            steps = [
                line.strip()
                for line in content.splitlines()
                if len(line.strip()) > 2 and not re.match(r'^\s*[\d\-\*•]\s*', line)  # avoid bullets
            ]
            steps = [s for s in steps if s.lower() not in ['steps:', 'step:']]
            return steps[:10]  # limit to 10 steps max
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}. Using fallback plan.")
            return [
                "Understand the task and requirements",
                "Plan the next steps",
                "Use coding or shell tools to make progress",
                "Test the result",
                "Review and iterate"
            ]

    def step(self, state) -> MessageAction:
        # Step 1: 如果尚未生成 TODO 列表，则生成
        if not self._planning_done:
            initial_task = ""
            with open("/instruction/task.md", "r") as f:
                initial_task = f.read()
            if initial_task:
                self.todo = self._generate_todo(initial_task)
                logger.info(f"[TodoCodeActAgent] Generated {len(self.todo)} steps: {self.todo}")
            else:
                logger.warning("No user task found in history. Using fallback step.")
                self.todo = ["Analyze the task", "Ask for clarification"]

            self._planning_done = True

        # Step 2: 构建当前步骤的指令
        if self.todo:
            current_step = self.todo[0]
            prefix_msg = (
                f"[Planner] You are currently working on this step:\n"
                f"{current_step}\n\n"
                f"Remaining steps: {len(self.todo) - 1}\n"
                "Please use tools to make progress. When this step is fully done, say exactly: 'STEP COMPLETED'."
            )
        else:
            return MessageAction(content="All steps completed.", source='agent')

        # 构造扩展的历史记录：插入规划提示作为新用户消息
        # 注意：不要修改原始 state.history，而是创建副本
        extended_history = [
            MessageAction(content=prefix_msg, source='user', keep_details=True)
        ] + list(state.history)

        # 创建新的 state（不污染原始）
        from pydantic import BaseModel
        if not isinstance(state, BaseModel):
            # 假设 state 是 dict-like
            new_state = state.copy()
            new_state['history'] = extended_history
        else:
            new_state = state.copy(update={'history': extended_history})

        # Step 3: 使用 CodeActAgent 执行
        try:
            action: MessageAction = self.codeact.step(new_state)
        except Exception as e:
            logger.error(f"Error during CodeAct execution: {e}")
            return MessageAction(content="Sorry, an error occurred while executing the step.", source='agent')

        # Step 4: 检查是否完成当前步骤
        if isinstance(action, MessageAction):
            if 'STEP COMPLETED' in action.content.strip():
                # 完成当前 step
                completed = self.todo.pop(0)
                logger.info(f"[TodoCodeActAgent] Step completed: {completed}")

                # 更新规划标志，以便后续步骤重新规划（可选）
                # 这里可以选择是否重新规划（比如每 3 步重新评估）

                if self.todo:
                    next_step = self.todo[0]
                    return MessageAction(
                        content=f"✅ Step completed: {completed}\n\n➡️  Continuing with: {next_step}",
                        source='agent',
                        keep_details=True
                    )
                else:
                    return MessageAction(
                        content="🎉 Task completed successfully.",
                        source='agent'
                    )

        # 返回正常响应（未完成当前 step）
        return action
