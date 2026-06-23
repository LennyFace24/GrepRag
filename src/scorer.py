"""
评分模块：基于 jieba 分词的 F1 评分
移植自 CLongEval 的 qa_f1_zh_score, 保持完全一致的计算逻辑
"""
import string
from collections import Counter

import jieba


# ── 中英文标点集合 ────────────────────────────────────
CN_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』"
    "【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
)
ALL_PUNCTUATION = set(string.punctuation + CN_PUNCTUATION)


def normalize_zh_answer(text: str) -> str:
    """去标点、去空白、转小写"""

    def white_space_fix(s: str) -> str:
        return "".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in ALL_PUNCTUATION)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_punc(lower(text)))


def f1_score(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    """token 级别的 F1 分数"""
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_zh_score(prediction: str, ground_truth: str) -> float:
    """
    CLongEval 原始评分函数:QA-F1-ZH

    1. 处理掉 LLM 自问自答的多余文本
    2. jieba 分词
    3. 每个 token 去标点/空白/小写
    4. 计算 F1

    Args:
        prediction: Agent 输出的答案
        ground_truth: 标准答案

    Returns:
        F1 分数 (0.0 ~ 1.0)
    """
    prediction = prediction.strip()

    # 处理一些 LLM 会在答案里自问自答的情况
    if "\n" in prediction:
        prediction = prediction.split("\n")[0]
    if "问题：" in prediction:
        pos_idx = prediction.find("问题：")
        prediction = prediction[:pos_idx]

    # jieba 分词
    pred_tokens = list(jieba.cut(prediction, cut_all=False))
    gt_tokens = list(jieba.cut(ground_truth, cut_all=False))

    # 每个 token 标准化
    pred_tokens = [normalize_zh_answer(t) for t in pred_tokens]
    gt_tokens = [normalize_zh_answer(t) for t in gt_tokens]

    # 去除空 token
    pred_tokens = [t for t in pred_tokens if len(t) > 0]
    gt_tokens = [t for t in gt_tokens if len(t) > 0]

    return f1_score(pred_tokens, gt_tokens)


def compute_accuracy(scores: list[float], threshold: float = 0.5) -> float:
    """将 F1 分数二值化为 accuracy (F1 >= threshold 视为正确)"""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores)
