"""
Agent 循环模块：管理 LLM 的检索决策循环
支持 Anthropic Claude 和 OpenAI GPT 两个 backend
"""
import json
import time
from pathlib import Path
from typing import Callable

from src.config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    ANTHROPIC_BASE_URL, OPENAI_BASE_URL,
    ANTHROPIC_MODEL, OPENAI_MODEL,
    MAX_AGENT_TURNS, AGENT_TEMPERATURE, AGENT_TIMEOUT,
    RETRY_COUNT,
)
from src.retrieval import get_tool_search_fn, get_tool_schema


# ──────────────────────────────────────────────────────
# 系统提示词 (中英双语版本)
# ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一个信息检索助手。一段多天对话记录存储在文件中，你需要通过搜索工具来查找信息并回答问题。

## 工作流程
1. 理解用户的问题，确定需要搜索什么信息
2. 使用提供的搜索工具在对话记录中查找相关内容
3. 根据搜索结果，判断是否已经找到答案
4. 一旦找到答案，立即用中文输出最终答案，停止搜索

## 重要规则
- 对话记录不会直接给你——你必须通过搜索工具来查找
- 搜索1-3次应该就足够了，不要进行不必要的额外搜索
- 如果搜索结果已经包含问题的答案，立即输出答案，不要继续搜索
- 答案应简洁：直接给出事实（如"《小王子》"），不要展开解释
- 如果2-3次搜索后仍然找不到相关内容，回答"无法从对话记录中找到答案"

当你准备好最终答案时，直接输出答案文本（不要调用工具）。"""


# ──────────────────────────────────────────────────────
# Agent 运行器
# ──────────────────────────────────────────────────────

class AgentRunner:
    """
    Agent 检索循环
    """

    def __init__(
        self,
        backend: str,                    # "anthropic" | "openai"
        tool_mode: str,                  # "grep" | "vector"
        model_name: str | None = None,
    ):
        self.backend = backend
        self.tool_mode = tool_mode

        if backend == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY 未设置")
            self._model = model_name or ANTHROPIC_MODEL
            import anthropic
            kwargs = {"api_key": ANTHROPIC_API_KEY, "timeout": AGENT_TIMEOUT}
            if ANTHROPIC_BASE_URL:
                kwargs["base_url"] = ANTHROPIC_BASE_URL
            self._client = anthropic.Anthropic(**kwargs)
        elif backend == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY 未设置")
            self._model = model_name or OPENAI_MODEL
            from openai import OpenAI
            kwargs = {"api_key": OPENAI_API_KEY, "timeout": AGENT_TIMEOUT}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            self._client = OpenAI(**kwargs)
        else:
            raise ValueError(f"不支持的 backend: {backend}。可选: anthropic, openai")

    def run(self, question: str, context_file: Path) -> str:
        """
        运行 Agent 循环

        Args:
            question: 用户问题
            context_file: 对话记录文件路径

        Returns:
            Agent 的最终答案 (纯文本)
        """
        # 初始化搜索函数
        search_fn = get_tool_search_fn(self.tool_mode, context_file)
        tool_schema = get_tool_schema(self.tool_mode, self.backend)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请回答以下问题：\n\n{question}"},
        ]

        for turn in range(MAX_AGENT_TURNS):
            try:
                if self.backend == "anthropic":
                    response = self._call_anthropic(messages, tool_schema)
                else:
                    response = self._call_openai(messages, tool_schema)
            except Exception as e:
                print(f"  [Agent] API 调用失败 (turn {turn + 1}): {e}")
                # 重试
                for retry in range(RETRY_COUNT):
                    try:
                        time.sleep(2 * (retry + 1))
                        if self.backend == "anthropic":
                            response = self._call_anthropic(messages, tool_schema)
                        else:
                            response = self._call_openai(messages, tool_schema)
                        break
                    except Exception:
                        if retry == RETRY_COUNT - 1:
                            return f"[ERROR] API 调用失败 {RETRY_COUNT} 次: {e}"
                else:
                    continue

            # 解析响应: 是 tool_call 还是 final_answer?
            if self.backend == "anthropic":
                is_tool_call, tool_args, text_response = self._parse_anthropic_response(response)
            else:
                is_tool_call, tool_args, text_response = self._parse_openai_response(response)

            if is_tool_call and tool_args:
                # 执行搜索工具
                query_str = tool_args.get("query", "")
                print(f"  [{self.tool_mode}] 搜索: '{query_str}'")

                search_result = search_fn(query_str)

                # 将工具调用和结果加入对话历史
                if self.backend == "anthropic":
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_args.get("_tool_use_id", ""),
                            "content": search_result,
                        }]
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_args.get("_tool_call_id", "call_1"),
                            "type": "function",
                            "function": {
                                "name": tool_schema["function"]["name"],
                                "arguments": json.dumps({"query": query_str}),
                            }
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_args.get("_tool_call_id", "call_1"),
                        "content": search_result,
                    })
            else:
                # 最终答案
                return text_response or "无法从对话记录中找到答案"

        # 超过最大轮次
        return "[TIMEOUT] 超过最大搜索轮次，未能得出结论。"

    # ── Anthropic API ──────────────────────────────────

    def _call_anthropic(self, messages: list[dict], tool_schema: dict):
        """调用 Claude API (支持 tool_use)"""
        # 构建 Anthropic 格式的 messages
        system_content = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                if isinstance(msg["content"], str):
                    api_messages.append({"role": "user", "content": msg["content"]})
                else:
                    api_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                api_messages.append({"role": "assistant", "content": msg["content"]})

        return self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=AGENT_TEMPERATURE,
            system=system_content,
            messages=api_messages,
            tools=[tool_schema],
        )

    def _parse_anthropic_response(self, response) -> tuple[bool, dict | None, str | None]:
        """解析 Claude 响应: 是 tool_use 还是纯文本?"""
        is_tool = False
        tool_args = None
        text_response = None

        for block in response.content:
            if block.type == "tool_use":
                is_tool = True
                tool_args = dict(block.input)
                tool_args["_tool_use_id"] = block.id
            elif block.type == "text":
                text_response = block.text

        return is_tool, tool_args, text_response

    # ── OpenAI API ─────────────────────────────────────

    def _call_openai(self, messages: list[dict], tool_schema: dict):
        """调用 OpenAI GPT API (支持 function calling)"""
        # 转换 tool schema 为 OpenAI 格式
        openai_tools = [{
            "type": "function",
            "function": {
                "name": tool_schema["function"]["name"],
                "description": tool_schema["function"]["description"],
                "parameters": tool_schema["function"]["parameters"],
            }
        }]

        # 构建 OpenAI 格式的 messages
        api_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                api_messages.append({"role": "system", "content": msg["content"]})
            elif role == "user":
                if isinstance(msg["content"], str):
                    api_messages.append({"role": "user", "content": msg["content"]})
                else:
                    api_messages.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                api_messages.append(msg)
            elif role == "tool":
                api_messages.append(msg)

        return self._client.chat.completions.create(
            model=self._model,
            messages=api_messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=AGENT_TEMPERATURE,
            max_tokens=1024,
        )

    def _parse_openai_response(self, response) -> tuple[bool, dict | None, str | None]:
        """解析 OpenAI 响应: 是 function_call 还是纯文本?"""
        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_args = json.loads(tool_call.function.arguments)
            tool_args["_tool_call_id"] = tool_call.id
            return True, tool_args, None
        else:
            return False, None, message.content
