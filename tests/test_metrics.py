"""metrics.py 的参数化单元测试。

覆盖全部 7 个指标函数以及 compute_all_metrics 统一入口，
使用 pytest.mark.parametrize 进行多场景参数化测试。
"""

from __future__ import annotations

import numpy as np
import pytest

# 注意：metrics.py 中存在重复函数定义（原文如此），
# Python 会使用文件中最后出现的定义。
from src.metrics import (
    compute_all_metrics,
    dcg_at_k,
    full_coverage_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ============================================================
# dcg_at_k
# ============================================================

class TestDcgAtK:
    """DCG@K 测试用例。"""

    @pytest.mark.parametrize(
        "relevances, k, expected",
        [
            # 空列表 -> 0
            ([], 5, 0.0),
            # 单个相关文档在首位: 1/log2(2) = 1.0
            ([1.0], 1, 1.0),
            # 二值相关性: [1, 0, 1]@2 -> 1/log2(2) + 0/log2(3) = 1.0
            ([1.0, 0.0, 1.0], 2, 1.0),
            # k 超出列表长度，截断到实际长度: 1/log2(2) + 0.5/log2(3)
            ([1.0, 0.5], 10, 1.0 / np.log2(2) + 0.5 / np.log2(3)),
        ],
    )
    def test_dcg(self, relevances, k, expected):
        result = dcg_at_k(relevances, k)
        assert result == pytest.approx(expected, abs=1e-9)


# ============================================================
# ndcg_at_k
# ============================================================

class TestNdcgAtK:
    """nDCG@K 测试用例。"""

    @pytest.mark.parametrize(
        "relevances, ideal_relevances, k, expected",
        [
            # 理想排序与实际一致 -> nDCG = 1.0
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1, 1.0),
            # 完美排序: ideal=[3,2,1], actual=[3,2,1] -> 1.0
            ([3.0, 2.0, 1.0], [3.0, 2.0, 1.0], 3, 1.0),
            # 逆转排序: ideal=[3,2,1], actual=[1,2,3] -> < 1.0
            # DCG([1,2,3], 3) = 1/log2(2) + 2/log2(3) + 3/log2(4)
            # IDCG([3,2,1], 3) = 3/log2(2) + 2/log2(3) + 1/log2(4)
            ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], 3, pytest.approx(0.79, abs=0.01)),
            # ideal 全 0 -> 0.0（避免除零）
            ([1.0, 1.0], [0.0, 0.0], 2, 0.0),
            # 空列表
            ([], [], 3, 0.0),
        ],
    )
    def test_ndcg(self, relevances, ideal_relevances, k, expected):
        result = ndcg_at_k(relevances, ideal_relevances, k)
        assert result == expected


# ============================================================
# mrr_at_k
# ============================================================

class TestMrrAtK:
    """MRR@K 测试用例。"""

    @pytest.mark.parametrize(
        "ranked_ids, relevant_ids, k, expected",
        [
            # 第一个就是相关文档 -> 1.0
            (["a", "b", "c"], {"a"}, 3, 1.0),
            # 第二个是相关文档 -> 0.5
            (["a", "b", "c"], {"b"}, 3, 0.5),
            # 无命中 -> 0.0
            (["a", "b", "c"], {"x"}, 3, 0.0),
            # k 截断导致无法命中
            (["a", "b", "c"], {"c"}, 2, 0.0),
            # 空排名列表
            ([], {"a"}, 5, 0.0),
        ],
    )
    def test_mrr(self, ranked_ids, relevant_ids, k, expected):
        result = mrr_at_k(ranked_ids, relevant_ids, k)
        assert result == pytest.approx(expected, abs=1e-9)


# ============================================================
# hit_at_k
# ============================================================

class TestHitAtK:
    """Hit@K 测试用例。"""

    @pytest.mark.parametrize(
        "ranked_ids, relevant_ids, k, expected",
        [
            # 命中
            (["a", "b", "c"], {"a"}, 1, 1.0),
            # k 截断后未命中
            (["a", "b", "c"], {"c"}, 1, 0.0),
            # 多个相关文档，命中其中一个
            (["a", "b", "c"], {"b", "c"}, 2, 1.0),
            # 空列表
            ([], {"a"}, 5, 0.0),
            # k=0
            (["a"], {"a"}, 0, 0.0),
        ],
    )
    def test_hit(self, ranked_ids, relevant_ids, k, expected):
        result = hit_at_k(ranked_ids, relevant_ids, k)
        assert result == pytest.approx(expected)


# ============================================================
# precision_at_k
# ============================================================

class TestPrecisionAtK:
    """Precision@K 测试用例。"""

    @pytest.mark.parametrize(
        "ranked_ids, relevant_ids, k, expected",
        [
            # 前 3 个中 2 个相关 -> 2/3
            (["a", "b", "c", "d"], {"a", "b"}, 3, pytest.approx(2 / 3)),
            # 全命中
            (["a", "b", "c"], {"a", "b", "c"}, 3, 1.0),
            # 零命中
            (["a", "b", "c"], {"x"}, 3, 0.0),
            # k=0 -> 0.0
            (["a"], {"a"}, 0, 0.0),
            # 空列表 k>0 -> 0.0
            ([], {"a"}, 3, 0.0),
        ],
    )
    def test_precision(self, ranked_ids, relevant_ids, k, expected):
        result = precision_at_k(ranked_ids, relevant_ids, k)
        assert result == expected


# ============================================================
# recall_at_k
# ============================================================

class TestRecallAtK:
    """Recall@K 测试用例。"""

    @pytest.mark.parametrize(
        "ranked_ids, relevant_ids, k, expected",
        [
            # 前 5 个覆盖了全部 2 个相关文档 -> 1.0
            (["a", "b", "c", "d", "e"], {"a", "b"}, 5, 1.0),
            # 覆盖一半
            (["a", "b", "c"], {"a", "x"}, 3, pytest.approx(0.5)),
            # 零覆盖
            (["a", "b"], {"x", "y"}, 2, 0.0),
            # 无相关文档 -> 0.0
            (["a", "b"], set(), 2, 0.0),
        ],
    )
    def test_recall(self, ranked_ids, relevant_ids, k, expected):
        result = recall_at_k(ranked_ids, relevant_ids, k)
        assert result == expected


# ============================================================
# full_coverage_at_k
# ============================================================

class TestFullCoverageAtK:
    """FullCoverage@K 测试用例（SkillRouter 自定义指标）。"""

    @pytest.mark.parametrize(
        "ranked_ids, required_ids, k, expected",
        [
            # 全覆盖
            (["a", "b", "c"], {"a", "b"}, 3, 1.0),
            # 部分覆盖 -> 0.0
            (["a", "b", "c"], {"a", "x"}, 3, 0.0),
            # k 不够大导致截断
            (["a", "b", "c"], {"a", "c"}, 2, 0.0),
            # 空 required_ids -> 1.0（定义如此）
            (["a", "b"], set(), 5, 1.0),
            # 排名列表为空但 required 也为空 -> 1.0
            ([], set(), 5, 1.0),
        ],
    )
    def test_full_coverage(self, ranked_ids, required_ids, k, expected):
        result = full_coverage_at_k(ranked_ids, required_ids, k)
        assert result == pytest.approx(expected)


# ============================================================
# compute_all_metrics — 统一入口
# ============================================================

class TestComputeAllMetrics:
    """compute_all_metrics 统一入口测试。"""

    EXPECTED_KEYS = {
        "nDCG@1", "nDCG@3", "nDCG@10",
        "Hit@1", "Precision@3",
        "MRR@10",
        "Recall@10", "Recall@20", "Recall@50",
        "FullCoverage@3", "FullCoverage@5", "FullCoverage@10",
    }

    def test_returns_all_12_keys(self):
        """验证返回字典包含全部 12 个指标键。"""
        ranked = ["s1", "s2", "s3"]
        gt = {"s1"}
        result = compute_all_metrics(ranked, gt)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_perfect_ranking(self):
        """排名第一即命中时，Hit@1=1, MRR@10=1, Recall@10=1。"""
        ranked = ["s1", "s2", "s3"]
        gt = {"s1"}
        result = compute_all_metrics(ranked, gt)
        assert result["Hit@1"] == pytest.approx(1.0)
        assert result["MRR@10"] == pytest.approx(1.0)
        assert result["Recall@10"] == pytest.approx(1.0)

    def test_no_overlap(self):
        """零重叠时 Hit@1=0, MRR@10=0, Recall=0。"""
        ranked = ["x1", "x2", "x3"]
        gt = {"s1"}
        result = compute_all_metrics(ranked, gt)
        assert result["Hit@1"] == pytest.approx(0.0)
        assert result["MRR@10"] == pytest.approx(0.0)
        assert result["Recall@10"] == pytest.approx(0.0)

    def test_multi_gt_full_coverage(self):
        """多个 ground truth，全部在 top-K 中时 FullCoverage@3=1。"""
        ranked = ["s1", "s2", "s3"]
        gt = {"s1", "s2"}
        result = compute_all_metrics(ranked, gt)
        assert result["FullCoverage@3"] == pytest.approx(1.0)

    def test_with_relevance_map(self):
        """使用分级相关性映射时，返回值合理且无异常。"""
        ranked = ["s1", "s2", "s3"]
        gt = {"s1", "s2"}
        relevance_map = {"s1": 3.0, "s2": 2.0, "s3": 0.0}
        result = compute_all_metrics(ranked, gt, relevance_map=relevance_map)
        # 12 个键均应存在
        assert set(result.keys()) == self.EXPECTED_KEYS
        # s1 排名第一且分值最高 -> nDCG@1 应为 1.0
        assert result["nDCG@1"] == pytest.approx(1.0)
