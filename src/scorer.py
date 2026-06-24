"""
基于 jieba 分词的 F1 评分
"""
import string
from collections import Counter
import jieba

_PUNC = set(string.punctuation + "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏.")

def _norm(s: str) -> str:
    return "".join(c for c in "".join(s.split()).lower() if c not in _PUNC)

def _f1(pred: list[str], gt: list[str]) -> float:
    same = sum((Counter(pred) & Counter(gt)).values())
    if not same: return 0.0
    p = same / len(pred)
    r = same / len(gt)
    return 2 * p * r / (p + r)

def score(prediction: str, ground_truth: str) -> float:
    """qa_f1_zh: jieba 分词 → 去标点 → F1"""
    pred = prediction.strip().split("\n")[0]
    if "问题：" in pred:
        pred = pred[:pred.find("问题：")]
    pt = [t for t in (_norm(t) for t in jieba.cut(pred)) if t]
    gt = [t for t in (_norm(t) for t in jieba.cut(ground_truth)) if t]
    return _f1(pt, gt) if pt and gt else 0.0

def acc(scores: list[float], threshold: float = 0.5) -> float:
    return sum(1 for s in scores if s >= threshold) / len(scores) if scores else 0.0
