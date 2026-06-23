# Grep vs RAG 中文检索率对比实验

## 1. 研究动机

英文中表达同一意思时词汇重叠度高（"I'm sad" → "I'm feeling down" 仍共享部分词汇），grep 容易命中。但中文中"同义异字"现象普遍（"我不开心" → "我情绪低落" → "我心情不太好"），同一事实在文本中的字面表达差异极大，grep 难以命中。RAG（语义检索）不受此限制。

**核心假设：** 中文场景下 grep 和 RAG 的检索效果差距显著大于英文场景。

## 2. 对标论文

arXiv:2605.15184 — *"Is Grep All You Need? How Agent Harnesses Reshape Agentic Search"*

- 在 LongMemEval 116 题上对比 grep vs vector retrieval
- 发现 inline 模式下 grep 在所有 harness-model 组合上均优于 vector
- 但论文明确指出其局限性：LongMemEval 的答案通常由 literal span 支持（英文），grep 的优势可能不适用于 paraphrase 密集的语言

本实验将其实验方法移植到中文场景，填补这一空白。

## 3. 实验设计

### 3.1 总体框架

```
自变量1: 检索方式 → grep-only vs RAG-only (组内)
自变量2: 语言 → 中文 vs 英文 (组间，目前先做中文)
自变量3: LLM Backend → Anthropic Claude vs OpenAI GPT (组内)
因变量:  F1 分数 / Accuracy
```

### 3.2 数据集

**中文：CLongEval / long_conversation_memory**

- 来源：HuggingFace (`zexuanqiu22/CLongEval`)
- 结构：用户与 AI 的跨多天对话记录，信息碎片化分布
- 问题类型：知识回忆（如"4月27日我推荐过什么书？"、"我喜欢哪种音乐？"、"我的名字是什么？"）
- 规模：small (~50 题, 1K-16K tokens)、medium (~50 题, 16K-50K tokens)、large (~50 题, 50K-100K tokens)
- 评估指标：基于 jieba 分词的 F1 分数

**英文（后续扩展）：LongMemEval 116 题子集**

### 3.3 Agent 架构

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  用户问题      │ ──→ │  LLM Agent       │ ──→ │  最终答案      │
│  (Question)   │     │  (Claude / GPT)   │     │  (Answer)     │
└──────────────┘     │                   │     └──────────────┘
                     │  有 Tool:          │
                     │  - grep_search()  │
                     │  - semantic_search│
                     │       ↓           │
                     │  检索对话文件       │
                     │       ↓           │
                     │  分析结果/继续搜索  │
                     └──────────────────┘
```

**Agent 循环：**
1. Agent 收到问题 + 系统提示词
2. Agent 决定调用搜索工具（带查询参数）
3. 工具在对话文本文件上执行检索 → 返回结果
4. Agent 分析结果 → 决定继续搜索或输出最终答案
5. 最多 8 轮搜索，超时强制结束

**关键设计：对话文本不直接给 Agent。Agent 必须通过检索工具来查找信息。** 这是本实验和 CLongEval 原始实验的核心区别。

### 3.4 检索工具实现

| | grep | RAG |
|---|---|---|
| 原理 | 正则表达式匹配 | Embedding + 余弦相似度 |
| 输入 | 关键词/正则 | 自然语言查询 |
| 输出 | 匹配行 + 上下文 | Top-K 最相似文本片段 |
| 分块 | N/A | 300 字符/chunk, 50 字符 overlap |
| 模型 | N/A | BAAI/bge-small-zh (可替换) |
| 优势 | 精确匹配、零成本 | 语义理解、同义表达 |
| 劣势 | 词汇不匹配时失效 | 需要 embedding 模型 |

### 3.5 实验配置矩阵

```
检索方式 × 数据集大小 × LLM Backend
────────────────────────────────────────
grep     × small/medium/large × anthropic
grep     × small/medium/large × openai
vector   × small/medium/large × anthropic
vector   × small/medium/large × openai

共计: 2 × 3 × 2 = 12 组
建议先跑 small 数据集（2 × 1 × 2 = 4 组）验证流程，再扩展到 full matrix。
```

### 3.6 评分方法

复用 CLongEval 的 `qa_f1_zh_score`：

1. jieba 分词
2. 去除中英文标点、空白
3. Token 级别 precision / recall / F1
4. 也可二值化为 accuracy（F1 ≥ 0.5 视为正确）

### 3.7 预期结果

| 场景 | grep 预期表现 | RAG 预期表现 |
|------|:---------:|:---------:|
| 英文，精确词汇匹配 | 高 | 高 |
| 英文，同义改写 | 中-低 | 高 |
| 中文，精确字词匹配 | 中-高 | 高 |
| 中文，同义异字 | 低 | 中-高 |

核心预期：**中文 grep 和 RAG 的性能差距 > 英文 grep 和 RAG 的性能差距**，尤其是在对话类口语化文本中。

## 4. 项目结构

```
grep/
├── data/                    # CLongEval 数据集
│   ├── small.jsonl
│   ├── medium.jsonl
│   └── large.jsonl
├── src/
│   ├── config.py            # 配置（API keys, 模型路径, 实验参数）
│   ├── data_loader.py       # 数据加载
│   ├── embedding.py         # Embedding 接口（BGE 默认 + Qwen 预留）
│   ├── retrieval.py         # Grep + Vector 检索工具
│   ├── agent.py             # Agent 循环（Anthropic + OpenAI）
│   ├── scorer.py            # F1 评分
│   ├── runner.py            # 实验运行器（断点续传）
│   └── report.py            # 结果汇总 + 表格
├── context_files/           # 对话上下文临时文件
├── results/                 # 实验结果输出
├── requirements.txt
└── EXPERIMENT_DESIGN.md     # 本文档
```

## 5. 运行方式

### 5.1 前置条件

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# 3. 确认数据就位
ls data/small.jsonl
```

### 5.2 运行实验

```bash
# 快速测试（前 5 题）
# 修改 src/config.py: LIMIT_QUESTIONS = 5, DATASET_SIZES = ["small"]

# 完整实验
python -m src.runner

# 查看结果
python -m src.report
```

## 6. 当前状态 (2026-06-23)

- [x] 项目结构搭建完成
- [x] 所有 Python 模块编写完成
- [x] 数据集下载就位
- [x] 数据加载验证通过
- [x] Grep 检索工具验证通过
- [ ] **需要配置有效的 API Key 并验证 Agent 端到端流程**
- [ ] RAG 检索工具需要首次下载 BGE 模型后验证
- [ ] 正式运行实验

## 7. 后续扩展方向

1. **英文对照组**：用 LongMemEval 跑相同的实验，得到中英文对比数据
2. **Hybrid 检索**：增加 grep + RAG 混合模式（论文中提到的 reciprocal rank fusion）
3. **Inline vs File-based**：对比两种工具结果交付方式
4. **噪声鲁棒性**：逐步增加干扰对话（对标论文 Experiment 2）
5. **多模型对比**：增加更多 LLM backend（如 Gemini, Qwen 等）
6. **文本类型扩展**：除了对话，加入新闻/百科等结构化文本
