"""
CLI Agent 运行器：通过 subprocess 调用 Claude Code / Codex CLI
对标论文的 provider-native CLI harness 实验方式
内置随机延迟和 429 退避，防止风控
"""
import subprocess
import shutil
import re
import random
import time
from pathlib import Path

from src.config import (
    CLI_AGENT_TIMEOUT, CLI_TEMPERATURE,
    RATE_LIMIT_DELAY_MIN, RATE_LIMIT_DELAY_MAX,
    RATE_LIMIT_BACKOFF, RATE_LIMIT_MAX_RETRIES,
)


# ── Prompts ────────────────────────────────────────────

def _build_grep_prompt(question: str, file_path: Path) -> str:
    """构建 grep 模式的 prompt"""
    return f"""搜索文件 {file_path} 中的对话记录来回答问题。
使用 grep 命令搜索。搜1-3次，找到答案后立即停止。

在回答的最后一行，输出：
FINAL_ANSWER: <答案内容>

问题：{question}"""


def _build_vector_prompt(question: str, file_path: Path, script_path: Path) -> str:
    """构建 vector/RAG 模式的 prompt"""
    return f"""搜索文件 {file_path} 中的对话记录来回答问题。
使用 python {script_path} "{file_path}" "查询" 进行语义搜索。搜1-3次，找到答案后立即停止。

在回答的最后一行，输出：
FINAL_ANSWER: <答案内容>

问题：{question}"""


# ── 速率控制 ────────────────────────────────────────────

def _rate_limit_delay():
    """每次请求前随机 sleep，防止连续请求触发风控"""
    delay = random.uniform(RATE_LIMIT_DELAY_MIN, RATE_LIMIT_DELAY_MAX)
    time.sleep(delay)


def _is_rate_limited(stdout: str, stderr: str) -> bool:
    """检测 CLI 输出中是否包含限流信号"""
    combined = (stdout + stderr).lower()
    markers = ["429", "rate limit", "too many requests", "rate exceeded",
               "quota exceeded", "try again later", "throttl"]
    return any(m in combined for m in markers)


# ── Output parsing ─────────────────────────────────────

def _extract_answer(output: str) -> str:
    """
    从 CLI 输出中提取最终答案
    优先匹配 "答案：xxx" 格式，否则取最后一段非空内容
    """
    # 匹配 "答案：xxx"
    match = re.search(r'答案[：:]\s*(.+?)(?:\n|$)', output)
    if match:
        return match.group(1).strip()

    # 回退：取最后一行有意义的内容
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    if lines:
        return lines[-1][:200]

    return "[EMPTY] CLI 没有产生有效输出"


# ── CLI Agent Runner ───────────────────────────────────

class CLIAgentRunner:
    """通过 subprocess 调用 CLI agent"""

    def __init__(self, backend: str, tool_mode: str, model_name: str | None = None):
        """
        Args:
            backend: "claude" | "codex"
            tool_mode: "grep" | "vector"
            model_name: 覆盖默认模型
        """
        self.backend = backend
        self.tool_mode = tool_mode
        self._model = model_name

        # 验证 CLI 可用
        if backend == "claude":
            self._cli_bin = shutil.which("claude")
            if not self._cli_bin:
                raise RuntimeError("claude CLI 未安装或不在 PATH 中")
        elif backend == "codex":
            self._cli_bin = shutil.which("codex")
            if not self._cli_bin:
                raise RuntimeError("codex CLI 未安装或不在 PATH 中")
        else:
            raise ValueError(f"不支持的 backend: {backend}。可选: claude, codex")

    def run(self, question: str, context_file: Path) -> tuple[str, str]:
        """运行 CLI agent, 返回 (答案, raw_output)"""
        if self.tool_mode == "grep":
            prompt = _build_grep_prompt(question, context_file)
        else:
            script_path = (Path(__file__).parent / "vector_search_cli.py").resolve()
            prompt = _build_vector_prompt(question, context_file, script_path)

        if self.backend == "claude":
            cmd = self._build_claude_cmd(prompt, context_file)
        else:
            cmd = self._build_codex_cmd(prompt, context_file)

        for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 2):
            # 随机延迟（首次也要延迟，防止连续请求）
            _rate_limit_delay()

            print(f"  [{self.backend}/{self.tool_mode}] 启动 CLI (第 {attempt} 次)...")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=CLI_AGENT_TIMEOUT,
                    cwd=str(Path(__file__).parent.parent),
                )
            except subprocess.TimeoutExpired:
                return f"[TIMEOUT] CLI 运行超时 ({CLI_AGENT_TIMEOUT}秒)", ""

            # 打印 stdout + stderr（不过滤，避免漏掉输出）
            if result.stdout.strip():
                print(f"  ── STDOUT ──")
                for line in result.stdout.split("\n"):
                    print(f"  | {line[:200]}")
            if result.stderr.strip():
                print(f"  ── STDERR ──")
                for line in result.stderr.split("\n"):
                    print(f"  ! {line[:200]}")

            # 检测 429 / rate limit
            if _is_rate_limited(result.stdout, result.stderr):
                wait = RATE_LIMIT_BACKOFF * (2 ** (attempt - 1))
                print(f"    ⚠ 触发速率限制, {wait:.0f}秒后重试...")
                time.sleep(wait)
                continue

            # 合并 stdout + stderr（某些 CLI 把实际输出写到 stderr）
            raw_output = result.stdout
            if not raw_output.strip() or result.returncode != 0:
                raw_output = (result.stdout + "\n" + result.stderr).strip()
            if not raw_output:
                raw_output = f"[EMPTY] CLI 退出码 {result.returncode}"

            return _extract_answer(raw_output), raw_output

        return f"[RATE_LIMITED] 已重试 {RATE_LIMIT_MAX_RETRIES} 次", ""

    def _build_claude_cmd(self, prompt: str, context_file: Path) -> list[str]:
        """构建 Claude Code CLI 命令"""
        context_dir = str(context_file.parent.resolve())

        cmd = [
            self._cli_bin,
            "-p", prompt,
            "--output-format", "text",
            "--add-dir", context_dir,
            "--max-budget-usd", "1",
        ]

        # 自动化标志：跳过权限确认
        cmd += ["--dangerously-skip-permissions"]

        # temperature（通过 JSON settings 注入）
        cmd += ["--settings", f'{{"temperature":{CLI_TEMPERATURE}}}']

        # 工具白名单
        if self.tool_mode == "grep":
            cmd += ["--allowedTools", f"Bash(grep:*)", f"Bash(cat:{context_file})"]
        else:
            cmd += ["--allowedTools", "Bash(python:*)"]

        # 模型
        if self._model:
            cmd += ["--model", self._model]

        return cmd

    def _build_codex_cmd(self, prompt: str, context_file: Path) -> list[str]:
        """构建 Codex CLI 命令"""
        context_dir = str(context_file.parent.resolve())

        cmd = [
            self._cli_bin, "exec", prompt,
            "--add-dir", context_dir,
            "--sandbox", "workspace-write",
            "--dangerously-bypass-approvals-and-sandbox",
        ]

        # 模型 + temperature
        cmd += ["-c", f"temperature={CLI_TEMPERATURE}"]
        if self._model:
            cmd += ["--model", self._model]

        return cmd
