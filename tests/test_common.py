"""common.py 的单元测试。

覆盖文本格式化函数（format_query / format_skill / format_rerank_prompt）
和 last_token_pool 在 CPU tensor 上的行为验证。
不依赖 GPU 或预训练模型权重。
"""

from __future__ import annotations

import pytest

# src.common 模块顶层 import torch，如果 torch DLL 不可用会导致整个模块加载失败。
# 先尝试导入，若失败则所有测试类会被条件跳过。
try:
    import torch

    from src.common import (
        QUERY_INSTRUCTION,
        RERANK_INSTRUCTION,
        aggregate,
        ensure_dir,
        format_query,
        format_rerank_prompt,
        format_skill,
        last_token_pool,
    )
    COMMON_AVAILABLE = True
except (ImportError, OSError):
    COMMON_AVAILABLE = False

requires_common = pytest.mark.skipif(not COMMON_AVAILABLE, reason="torch/common 模块不可用")


# ============================================================
# format_query
# ============================================================

@requires_common
class TestFormatQuery:
    """format_query 测试。"""

    def test_prefix_instruction(self):
        """返回字符串以 QUERY_INSTRUCTION 为前缀。"""
        result = format_query("test task")
        assert result.startswith(QUERY_INSTRUCTION)

    def test_appends_query(self):
        """原始查询内容应跟在指令之后。"""
        result = format_query("fix the bug")
        assert result.endswith("fix the bug")

    def test_truncation(self):
        """超出 max_len 的部分应被截断。"""
        long_query = "x" * 3000
        result = format_query(long_query, max_len=100)
        # 截断后查询部分长度为 100
        query_part = result[len(QUERY_INSTRUCTION):]
        assert len(query_part) == 100

    def test_default_max_len(self):
        """默认 max_len=1500 应生效。"""
        query = "a" * 2000
        result = format_query(query)
        query_part = result[len(QUERY_INSTRUCTION):]
        assert len(query_part) == 1500


# ============================================================
# format_skill
# ============================================================

@requires_common
class TestFormatSkill:
    """format_skill 测试。"""

    def test_basic_format(self):
        """返回格式为 'name | description | body'。"""
        skill = {"name": "MySkill", "description": "A tool", "body": "def run(): pass"}
        result = format_skill(skill)
        assert result == "MySkill | A tool | def run(): pass"

    def test_missing_fields(self):
        """缺少字段时应使用空字符串代替。"""
        skill = {"name": "OnlyName"}
        result = format_skill(skill)
        assert result == "OnlyName |  | "

    def test_desc_truncation(self):
        """description 超过 desc_max 应被截断。"""
        skill = {"name": "n", "description": "d" * 500, "body": "b"}
        result = format_skill(skill, desc_max=100)
        assert "| " + "d" * 100 + " |" in result

    def test_body_truncation(self):
        """body 超过 body_max 应被截断。"""
        skill = {"name": "n", "description": "d", "body": "b" * 5000}
        result = format_skill(skill, body_max=200)
        # body 部分最多 200 字符
        parts = result.split(" | ")
        assert len(parts[2]) == 200


# ============================================================
# format_rerank_prompt
# ============================================================

@requires_common
class TestFormatRerankPrompt:
    """format_rerank_prompt 测试。"""

    def test_flat_full_format(self):
        """flat-full 格式：文档部分为 name | desc | body。"""
        result = format_rerank_prompt("n", "d", "b", "q", prompt_format="flat-full")
        assert RERANK_INSTRUCTION in result
        assert "<Query>: q" in result
        assert "<Document>: n | d | b" in result

    def test_flat_nd_format(self):
        """flat-nd 格式：文档部分不含 body。"""
        result = format_rerank_prompt("n", "d", "b", "q", prompt_format="flat-nd")
        assert "<Document>: n | d" in result
        assert "b" not in result

    def test_struct_format(self):
        """struct 格式：使用 XML 标签包裹各字段。"""
        result = format_rerank_prompt("n", "d", "b", "q", prompt_format="struct")
        assert "<Name>: n" in result
        assert "<Description>: d" in result
        assert "<Body>: b" in result
        assert "<Skill>:" in result

    def test_unknown_format_raises(self):
        """未知 prompt_format 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="Unknown prompt_format"):
            format_rerank_prompt("n", "d", "b", "q", prompt_format="invalid")

    def test_truncation_desc_body(self):
        """desc 和 body 超长时应被截断。"""
        result = format_rerank_prompt(
            "n", "d" * 1000, "b" * 5000, "q",
            prompt_format="flat-full", desc_max=50, body_max=100,
        )
        # desc 截断到 50 字符
        assert "d" * 50 in result
        assert "d" * 51 not in result


# ============================================================
# last_token_pool — CPU tensor 测试
# ============================================================

@requires_common
class TestLastTokenPool:
    """last_token_pool 函数在 CPU tensor 上的测试。"""

    def test_right_padding_returns_last_non_pad(self):
        """右填充场景：应返回每个序列最后一个非 padding token 的隐藏状态。"""
        # 序列 1: 3 个有效 token，序列 2: 1 个有效 token
        hidden_states = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.0, 0.0, 0.0]],
            [[10.0, 20.0, 30.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ])
        result = last_token_pool(hidden_states, attention_mask)
        # 序列 1: 最后有效位置 index=2 -> [7,8,9]
        # 序列 2: 最后有效位置 index=0 -> [10,20,30]
        expected = torch.tensor([[7.0, 8.0, 9.0], [10.0, 20.0, 30.0]])
        assert torch.allclose(result, expected)

    def test_left_padding_returns_last_position(self):
        """左填充场景（attention_mask 全 1）：应返回最后一个位置的隐藏状态。"""
        hidden_states = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ])
        # 全 1 -> 判定为左填充
        attention_mask = torch.tensor([
            [1, 1, 1],
            [1, 1, 1],
        ])
        result = last_token_pool(hidden_states, attention_mask)
        expected = torch.tensor([[5.0, 6.0], [11.0, 12.0]])
        assert torch.allclose(result, expected)

    def test_single_token_sequence(self):
        """单 token 序列：直接返回该 token 的隐藏状态。"""
        hidden_states = torch.tensor([[[42.0, 43.0]]])
        attention_mask = torch.tensor([[1]])
        result = last_token_pool(hidden_states, attention_mask)
        expected = torch.tensor([[42.0, 43.0]])
        assert torch.allclose(result, expected)

    def test_output_shape(self):
        """输出形状应为 (batch_size, hidden_dim)。"""
        hidden_states = torch.randn(3, 5, 8)
        attention_mask = torch.ones(3, 5, dtype=torch.long)
        result = last_token_pool(hidden_states, attention_mask)
        assert result.shape == (3, 8)


# ============================================================
# aggregate
# ============================================================

@requires_common
class TestAggregate:
    """aggregate 均值聚合函数测试。"""

    def test_basic_aggregation(self):
        """验证均值聚合正确性。"""
        metrics_list = [
            {"nDCG@1": 1.0, "Hit@1": 1.0},
            {"nDCG@1": 0.0, "Hit@1": 0.0},
        ]
        result = aggregate(metrics_list)
        assert result["nDCG@1"] == pytest.approx(0.5)
        assert result["Hit@1"] == pytest.approx(0.5)
        assert result["count"] == 2

    def test_empty_list(self):
        """空列表应返回空字典。"""
        assert aggregate([]) == {}


# ============================================================
# ensure_dir
# ============================================================

@requires_common
class TestEnsureDir:
    """ensure_dir 目录创建测试。"""

    def test_creates_nested_dir(self, tmp_path):
        """应能递归创建嵌套目录。"""
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert result.exists()
        assert result.is_dir()

    def test_existing_dir_no_error(self, tmp_path):
        """目录已存在时不应报错。"""
        result = ensure_dir(tmp_path)
        assert result == tmp_path
