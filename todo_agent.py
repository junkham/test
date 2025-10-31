from __future__ import annotations
import json, re
from pathlib import Path
from typing import List

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.agent import Agent
from openhands.core.config import LLMConfig
from openhands.events.action import Action, MessageAction
from openhands.events.observation import Observation
from openhands.llm.llm import LLM

# ---------- 工具：勾掉/更新 TODO ----------
class UpdateTodoAction(Action):
    action: str = "update_todo"
    todo_list: List[str]
    done_index: int                # 刚完成的序号（0-base）

    def run(self, controller):
        self.todo_list.pop(self.done_index)
        return MessageAction(content="TODO updated", metadata={"remaining": self.todo_list})


# ---------- 新 Agent：规划 + CodeAct ----------
class TodoCodeActAgent(Agent):
    """外层规划器，内层仍用原 CodeActAgent 干活"""
    def __init__(self, llm_config: LLMConfig, config=None):
        super().__init__(llm_config, config)
        self.llm = LLM(llm_config)
        self.codeact = CodeActAgent(self.llm, config)   # 真正干活的
        self.todo: List[str] = []

    # ---- 1. 一次性生成 TODO ----
    def _make_todo(self, history) -> List[str]:
        prompt = (
            "You are a planner. Read the task below and output a concise todo list "
            "（每行一个，简短动词开头，不含编号）:\n\n"
            f"{history[0].content}\n\n"
            "TODO:"
        )
        resp = self.llm.completion(prompt)
        return [line.strip("- ") for line in resp.splitlines() if line.strip()]

    # ---- 2. 每轮把剩余子任务写进 prompt ----
    def _wrap_prompt(self, history):
        if not self.todo:
            return history
        todo_txt = "\n".join(f"{i}. {t}" for i, t in enumerate(self.todo, 1))
        header = MessageAction(
            content=f"[Planner] 剩余子任务：\n{todo_txt}\n请完成第 1 项，然后返回简短结果。",
            source="user",
        )
        return [header] + history

    # ---- 3. 主入口 ----
    def step(self, state) -> Action:
        # 首次调用：生成 TODO
        if not self.todo:
            self.todo = self._make_todo(state.history)
            logger.info("[TodoCodeAct] 生成 %d 项子任务", len(self.todo))

        # 把剩余任务注入上下文
        wrapped_history = self._wrap_prompt(state.history)

        # 让 CodeActAgent 干活
        action = self.codeact.step(state.copy(update={"history": wrapped_history}))

        # 如果 CodeActAgent 认为做完了，就勾掉当前子任务
        if isinstance(action, MessageAction) and action.source == "agent":
            if self.todo:
                return UpdateTodoAction(todo_list=self.todo.copy(), done_index=0)
            else:
                return action   # 真正结束
        return action

    # ---- 4. 观察 TODO 更新 ----
    def on_observation(self, obs: Observation):
        if isinstance(obs, MessageAction) and obs.metadata.get("remaining") is not None:
            self.todo = obs.metadata["remaining"]
            logger.info("[TodoCodeAct] 剩余 %d 项", len(self.todo))
