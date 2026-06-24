"""
CLI Agent：通过 subprocess 调用 Claude Code / Codex CLI
"""
import subprocess, shutil, re, random, time
from pathlib import Path
from src.config import CLI_TIMEOUT, CLI_TEMP, DELAY_MIN, DELAY_MAX, RL_BACKOFF, RL_RETRIES

# ── 速率控制 ────────────────────────────────────────────

def _jitter():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

def _is_rate_limited(out: str) -> bool:
    return any(m in out.lower() for m in
        ["429", "rate limit", "too many requests", "rate exceeded",
         "quota exceeded", "try again later", "throttl"])

# ── 答案提取 ────────────────────────────────────────────

def _extract_answer(output: str) -> str:
    """匹配 FINAL_ANSWER: xxx 或 答案：xxx，否则取最后一行"""
    for pat in [r'FINAL_ANSWER:\s*(.+)', r'答案[：:]\s*(.+)']:
        m = re.search(pat, output)
        if m:
            ans = m.group(1).strip()
            if ans not in ("你的答案", "答案内容", "<答案内容>"):  # 排除 prompt 模板
                return ans
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    return lines[-1][:200] if lines else "[EMPTY]"

# ── Prompt ──────────────────────────────────────────────

def _prompt(question: str, file_path: Path, tool_mode: str) -> str:
    if tool_mode == "grep":
        tool = f"grep 搜索文件 {file_path}"
    else:
        script = (Path(__file__).parent / "vector_search_cli.py").resolve()
        tool = f"python {script} \"{file_path}\" \"查询\""
    return f"在 {file_path} 中搜索对话记录来回答问题。用 {tool}。\n最后一行输出: FINAL_ANSWER: <答案>\n问题: {question}"

# ── CLI Runner ──────────────────────────────────────────

_ERROR_MARKERS = ["cannot be used with root", "permission denied", "command not found",
                  "usage:", "api key", "authentication", "unauthorized"]

class CLIAgentRunner:
    def __init__(self, backend: str, tool_mode: str, model_name: str | None = None):
        self.backend = backend
        self.tool_mode = tool_mode
        self._model = model_name
        self._bin = shutil.which(backend)
        if not self._bin:
            raise RuntimeError(f"{backend} CLI 未安装")

    def run(self, question: str, ctx_file: Path) -> tuple[str, str]:
        prompt = _prompt(question, ctx_file, self.tool_mode)
        cmd = self._cmd(prompt, ctx_file)

        for attempt in range(1, RL_RETRIES + 2):
            _jitter()
            print(f"  [{self.backend}/{self.tool_mode}] #{attempt}...")

            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=CLI_TIMEOUT,
                                   cwd=str(Path(__file__).parent.parent))
            except subprocess.TimeoutExpired:
                return f"[TIMEOUT] {CLI_TIMEOUT}s", ""

            raw = (r.stdout + "\n" + r.stderr).strip()
            for line in raw.split("\n"):
                print(f"  | {line[:200]}")
            if not raw:
                raw = f"[EMPTY] exit={r.returncode}"

            if _is_rate_limited(raw):
                wait = RL_BACKOFF * (2 ** (attempt - 1))
                print(f"    ⚠ rate limited, {wait:.0f}s 后重试")
                time.sleep(wait)
                continue

            ans = _extract_answer(raw)
            if any(m in ans.lower() for m in _ERROR_MARKERS):
                return f"[CLI_ERROR] {ans[:100]}", raw
            return ans, raw

        return f"[RATE_LIMITED] 重试{RL_RETRIES}次", ""

    def _cmd(self, prompt: str, ctx_file: Path) -> list[str]:
        d = str(ctx_file.parent.resolve())
        if self.backend == "claude":
            c = [self._bin, "-p", prompt, "--output-format", "text",
                 "--add-dir", d, "--max-budget-usd", "1",
                 "--settings", f'{{"temperature":{CLI_TEMP}}}']
            if self.tool_mode == "grep":
                c += ["--allowedTools", f"Bash(grep:*)", f"Bash(cat:{ctx_file})"]
            else:
                c += ["--allowedTools", "Bash(python:*)"]
            if self._model:
                c += ["--model", self._model]
        else:
            c = [self._bin, "exec", prompt, "--add-dir", d,
                 "--sandbox", "workspace-write",
                 "--dangerously-bypass-approvals-and-sandbox",
                 "-c", f"temperature={CLI_TEMP}"]
            if self._model:
                c += ["--model", self._model]
        return c
