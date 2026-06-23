"""
CLI Agent 运行器：通过 subprocess 调用 Claude Code / Codex CLI
对标论文的 provider-native CLI harness 实验方式
"""
import subprocess
import shutil
import re
from pathlib import Path

from src.config import (
    CLI_AGENT_TIMEOUT,
)


# ── Prompts ────────────────────────────────────────────

def _build_grep_prompt(question: str, file_path: Path) -> str:
    """构建 grep 模式的 prompt"""
    return f"""一段多天对话记录存储在文件 {file_path} 中。

你只能通过 grep 命令搜索这个文件来查找信息。例如：
  grep "关键词" {file_path}
  grep -C 2 "正则" {file_path}

搜索规则：
- 优先用问题中出现的原词搜索
- 如果搜不到，尝试拆分关键词、用更短的词
- 搜索1-3次即可，找到信息后直接给出答案
- 不要用 cat 读取整个文件，文件太长

**在你回答的最后一行，单独写一行 "答案：你的答案"，除此之外不要输出其他解释。**

问题：{question}"""


def _build_vector_prompt(question: str, file_path: Path, script_path: Path) -> str:
    """构建 vector/RAG 模式的 prompt"""
    return f"""一段多天对话记录存储在文件 {file_path} 中。

你只能通过以下命令进行语义搜索（它用向量相似度查找相关内容）：
  python {script_path} "{file_path}" "你的查询"

搜索规则：
- 用自然语言描述你想查找的信息
- 如果第一次搜索结果不够，可以换一种表述再搜一次
- 搜索1-3次即可，找到信息后直接给出答案

**在你回答的最后一行，单独写一行 "答案：你的答案"，除此之外不要输出其他解释。**

问题：{question}"""


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

    def run(self, question: str, context_file: Path) -> str:
        """运行 CLI agent 并返回答案"""
        # 构建 prompt
        if self.tool_mode == "grep":
            prompt = _build_grep_prompt(question, context_file)
        else:
            script_path = (Path(__file__).parent / "vector_search_cli.py").resolve()
            prompt = _build_vector_prompt(question, context_file, script_path)

        # 构建命令
        if self.backend == "claude":
            cmd = self._build_claude_cmd(prompt, context_file)
        else:
            cmd = self._build_codex_cmd(prompt, context_file)

        print(f"  [{self.backend}/{self.tool_mode}] 启动 CLI...")
        print(f"  命令: {' '.join(cmd[:6])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=CLI_AGENT_TIMEOUT,
                cwd=str(Path(__file__).parent.parent),  # 项目根目录
            )
        except subprocess.TimeoutExpired:
            return f"[TIMEOUT] CLI 运行超时 ({AGENT_TIMEOUT * 3}秒)"

        output = result.stdout
        if result.returncode != 0 and not output.strip():
            output = result.stderr or f"[ERROR] CLI 退出码 {result.returncode}"

        return _extract_answer(output)

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

        # 模型（留空则用 Codex CLI 自身配置的默认模型）
        if self._model:
            cmd += ["--model", self._model]

        return cmd
