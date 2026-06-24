# Grep vs RAG 中文检索率对比实验

对标 arXiv:2605.15184（"Is Grep All You Need?"），将实验方法移植到中文场景，研究中文"同义异字"现象对 grep 和 RAG 检索效果的影响。

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### 2. 配置实验

```bash
cp .env.example .env
# 编辑 .env 填入你的配置
```

**`.env` 关键配置说明：**

```bash
# ── CLI Agent（必选）──────────────────────────────────────
BACKENDS=claude                    # claude / codex / claude,codex
CLI_TEMPERATURE=0.1

# ── Embedding（vector 模式需要）───────────────────────────
EMBED_PROVIDER=bge                 # bge(本地) / qwen(API)
# qwen 模式需要：
# QWEN_EMBED_KEY=你的key
# QWEN_EMBED_MODEL=Qwen3-Embedding-0.6B

# ── 实验矩阵 ─────────────────────────────────────────────
DATASET_SIZES=small                # small / medium / large
TOOL_MODES=grep,vector             # grep / vector / grep,vector

# ── 运行控制 ─────────────────────────────────────────────
LIMIT_QUESTIONS=                   # 留空=全部, 填数字=只跑前N题(调试)
MAX_WORKERS=1                      # 1=顺序, 4=4并发
```

### 3. 准备数据

从 HuggingFace 下载 CLongEval 数据集：

```bash
pip install huggingface-hub
huggingface-cli download zexuanqiu22/CLongEval --repo-type dataset --local-dir ./tmp
cp tmp/long_conversation_memory/small.jsonl data/
cp tmp/long_conversation_memory/medium.jsonl data/
cp tmp/long_conversation_memory/large.jsonl data/
rm -rf tmp
```

### 4. 运行实验

```bash
# 调试：只跑 3 题
# .env 里设置 LIMIT_QUESTIONS=3
python -m src.runner

# 正式运行
# .env 里设置 LIMIT_QUESTIONS=  (留空)
python -m src.runner
```

运行时会看到：
```
==================================================
  实验: small | grep | claude
==================================================
  已完成: 0, 剩余: 358

  [1/358] Q: 4月27日，我和你推荐过一本书，书名是什么？...
  [claude/grep] #1...
  | FINAL_ANSWER: 《小王子》
    A: 《小王子》...
    耗时: 9.2s
```

### 5. 查看结果

```bash
# 终端表格
python -m src.report

# 终端图表 + 每题对比
python -m src.visualize

# matplotlib 图表 (需要 pip install matplotlib)
python -m src.visualize
# → 生成 results/comparison.png
```

## 项目结构

```
├── data/                    # 数据集 (不提交 git)
├── src/
│   ├── config.py            # 配置 (.env 读取)
│   ├── data_loader.py       # 数据加载
│   ├── embedding.py         # Embedding (BGE/Qwen)
│   ├── cli_agent.py         # CLI Agent (claude/codex)
│   ├── vector_search_cli.py # 向量搜索 CLI (agent 调用)
│   ├── scorer.py            # F1 评分
│   ├── runner.py            # 实验运行器
│   ├── report.py            # 结果报告
│   └── visualize.py         # 可视化
├── results/                 # 实验结果
├── context_files/           # 对话文本临时文件
├── .env.example             # 配置模板
└── requirements.txt
```

## 实验原理

```
┌─────────────┐     ┌───────────────────┐     ┌──────────┐
│   问题       │ ──→ │  CLI Agent        │ ──→ │  答案     │
│             │     │  (Claude/Codex)    │     │          │
└─────────────┘     │                   │     └──────────┘
                    │  搜索工具:         │
                    │  - grep (grep)    │
                    │  - semantic (RAG)  │
                    │        ↓           │
                    │  搜索对话文件       │
                    └───────────────────┘
                            ↓
                    F1 评分 vs 标准答案
```

## 参考

- [arXiv:2605.15184](https://arxiv.org/abs/2605.15184) — Is Grep All You Need?
- [CLongEval](https://github.com/zexuanqiu/CLongEval) — 中文长文本评测基准
