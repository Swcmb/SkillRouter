"""信息检索评估指标模块。

提供 nDCG、MRR、Hit、Precision、Recall、FullCoverage 等常用 IR 指标的计算函数，
以及统一的 compute_all_metrics 入口，用于 SkillRouter 的检索与重排评估。
"""

from __future__ import annotations

import numpy as np


def dcg_at_k(relevances: list[float], k: int) -> float:
    """计算 Discounted Cumulative Gain (DCG@K)。

    以 log2(i+2) 作为折扣因子，对前 K 个位置的相关性分数进行加权求和。

    Args:
        relevances: 按排名顺序排列的相关性分数列表。
        k: 截断位置，仅考虑前 K 个结果。

    Returns:
        DCG@K 的浮点数值。
    """
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[float], ideal_relevances: list[float], k: int) -> float:
    """计算归一化折损累积增益 (nDCG@K)。

    将实际 DCG 除以理想排序下的 DCG 进行归一化，取值范围 [0, 1]。

    Args:
        relevances: 实际排序结果的相关性分数列表。
        ideal_relevances: 理想排序（按相关性降序）的分数列表。
        k: 截断位置。

    Returns:
        nDCG@K 的浮点数值，理想 DCG 为零时返回 0.0。
    """
    dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """计算前 K 个结果的平均倒数排名 (MRR@K)。

    找到排名列表中第一个命中的相关文档，返回其排名的倒数。

    Args:
        ranked_ids: 按排名顺序排列的文档 ID 列表。
        relevant_ids: 相关文档 ID 集合。
        k: 截断位置。

    Returns:
        第一个相关文档排名的倒数，未命中则返回 0.0。
    """
    for i, rid in enumerate(ranked_ids[:k]):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """判断前 K 个结果中是否包含至少一个相关文档 (Hit@K)。

    Args:
        ranked_ids: 按排名顺序排列的文档 ID 列表。
        relevant_ids: 相关文档 ID 集合。
        k: 截断位置。

    Returns:
        命中返回 1.0，未命中返回 0.0。
    """
    return 1.0 if any(rid in relevant_ids for rid in ranked_ids[:k]) else 0.0


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """计算前 K 个结果的精确率 (Precision@K)。

    Args:
        ranked_ids: 按排名顺序排列的文档 ID 列表。
        relevant_ids: 相关文档 ID 集合。
        k: 截断位置。

    Returns:
        前 K 个结果中相关文档的比例，k 为 0 时返回 0.0。
    """
    hits = sum(1 for rid in ranked_ids[:k] if rid in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """计算前 K 个结果的召回率 (Recall@K)。

    Args:
        ranked_ids: 按排名顺序排列的文档 ID 列表。
        relevant_ids: 相关文档 ID 集合。
        k: 截断位置。

    Returns:
        前 K 个结果覆盖的相关文档占全部相关文档的比例，
        无相关文档时返回 0.0。
    """
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def full_coverage_at_k(ranked_ids: list[str], required_ids: set[str], k: int) -> float:
    """判断前 K 个结果是否覆盖全部必需文档 (FullCoverage@K)。

    这是 SkillRouter 提出的自定义指标，用于衡量多技能任务场景下
    检索结果是否包含所有必需技能。

    Args:
        ranked_ids: 按排名顺序排列的文档 ID 列表。
        required_ids: 必需文档 ID 集合（通常为某个任务的全部 ground truth 技能）。
        k: 截断位置。

    Returns:
        全部必需文档均被覆盖返回 1.0，否则返回 0.0；
        无必需文档时返回 1.0。
    """
    if not required_ids:
        return 1.0
    return 1.0 if required_ids.issubset(set(ranked_ids[:k])) else 0.0


def compute_all_metrics(
    ranked_ids: list[str],
    gt_skill_ids: set[str],
    relevance_map: dict[str, float] | None = None,
) -> dict[str, float]:
    """统一计算所有检索评估指标。

    根据排名列表和 ground truth 计算 12 项 IR 指标，支持二值相关性和分级相关性
    两种模式。

    Args:
        ranked_ids: 按排名顺序排列的文档 ID 列表。
        gt_skill_ids: ground truth 相关文档 ID 集合。
        relevance_map: 可选的分级相关性映射 {skill_id: relevance_score}。
            提供时使用分级 nDCG；否则退化为二值相关性。

    Returns:
        包含 12 项指标的字典，键名如 "nDCG@1"、"Hit@1"、"FullCoverage@10" 等。
    """
    # 根据 relevance_map 构建相关性序列
    if relevance_map:
        relevances = [float(relevance_map.get(rid, 0)) for rid in ranked_ids]
        all_relevance_values = list(relevance_map.values())
    else:
        relevances = [1.0 if rid in gt_skill_ids else 0.0 for rid in ranked_ids]
        all_relevance_values = [1.0] * len(gt_skill_ids) + [0.0] * max(0, len(ranked_ids) - len(gt_skill_ids))

    # 按截断位置分组调用，避免重复截断 ranked_ids[:k]
    results: dict[str, float] = {}

    # nDCG 指标（依赖 relevance 向量）
    for k in (1, 3, 10):
        results[f"nDCG@{k}"] = ndcg_at_k(relevances, all_relevance_values, k)

    # 基于 ID 匹配的指标
    for k in (1,):
        results[f"Hit@{k}"] = hit_at_k(ranked_ids, gt_skill_ids, k)
    for k in (3,):
        results[f"Precision@{k}"] = precision_at_k(ranked_ids, gt_skill_ids, k)
    for k in (10,):
        results[f"MRR@{k}"] = mrr_at_k(ranked_ids, gt_skill_ids, k)
    for k in (10, 20, 50):
        results[f"Recall@{k}"] = recall_at_k(ranked_ids, gt_skill_ids, k)
    for k in (3, 5, 10):
        results[f"FullCoverage@{k}"] = full_coverage_at_k(ranked_ids, gt_skill_ids, k)

    return results
