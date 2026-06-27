"""data_io.py 的单元测试。

覆盖 JSONL / gzip JSONL 的读取、流式迭代、行计数，
以及 iter_jsonl_paths 的路径发现逻辑。
使用 pytest tmp_path fixture 管理临时文件。
"""

from __future__ import annotations

import gzip
import json

import pytest

from src.data_io import count_jsonl, iter_jsonl_paths, load_jsonl, open_text, stream_jsonl

# ============================================================
# 辅助工厂：创建临时 JSONL / .jsonl.gz 文件
# ============================================================

def _write_jsonl(path, records: list[dict]):
    """将记录列表写入 .jsonl 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_gz_jsonl(path, records: list[dict]):
    """将记录列表写入 .jsonl.gz 文件。"""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# iter_jsonl_paths
# ============================================================

class TestIterJsonlPaths:
    """iter_jsonl_paths 路径发现测试。"""

    def test_single_jsonl_file(self, tmp_path):
        """单个 .jsonl 文件应返回包含自身的列表。"""
        f = tmp_path / "data.jsonl"
        f.write_text("{}\n")
        result = iter_jsonl_paths(f)
        assert result == [f]

    def test_single_gz_file(self, tmp_path):
        """单个 .jsonl.gz 文件应返回包含自身的列表。"""
        f = tmp_path / "data.jsonl.gz"
        _write_gz_jsonl(f, [{"id": 1}])
        result = iter_jsonl_paths(f)
        assert result == [f]

    def test_directory_sorted(self, tmp_path):
        """目录下的文件应按名称排序返回。"""
        for name in ["b.jsonl", "a.jsonl", "c.jsonl.gz"]:
            (tmp_path / name).write_text("{}\n")
        result = iter_jsonl_paths(tmp_path)
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_nonexistent_raises(self, tmp_path):
        """不存在的路径应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            iter_jsonl_paths(tmp_path / "no_such_file.jsonl")

    def test_non_jsonl_file_excluded(self, tmp_path):
        """目录中非 .jsonl/.jsonl.gz 文件应被排除。"""
        (tmp_path / "data.jsonl").write_text("{}\n")
        (tmp_path / "readme.txt").write_text("hello\n")
        result = iter_jsonl_paths(tmp_path)
        assert len(result) == 1
        assert result[0].name == "data.jsonl"


# ============================================================
# load_jsonl
# ============================================================

class TestLoadJsonl:
    """load_jsonl 读取测试。"""

    def test_basic_load(self, tmp_path):
        """应正确读取所有记录。"""
        records = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        f = tmp_path / "test.jsonl"
        _write_jsonl(f, records)
        result = load_jsonl(f)
        assert result == records

    def test_gzip_load(self, tmp_path):
        """应能正确读取 .jsonl.gz 格式。"""
        records = [{"key": "value"}]
        f = tmp_path / "test.jsonl.gz"
        _write_gz_jsonl(f, records)
        result = load_jsonl(f)
        assert result == records

    def test_skips_blank_lines(self, tmp_path):
        """空行应被跳过。"""
        f = tmp_path / "blank.jsonl"
        with open(f, "w", encoding="utf-8") as fp:
            fp.write('{"a": 1}\n')
            fp.write('\n')
            fp.write('   \n')
            fp.write('{"b": 2}\n')
        result = load_jsonl(f)
        assert len(result) == 2

    def test_directory_load(self, tmp_path):
        """从目录加载时，应合并所有 shard 的记录。"""
        _write_jsonl(tmp_path / "shard_0.jsonl", [{"id": "a"}])
        _write_jsonl(tmp_path / "shard_1.jsonl", [{"id": "b"}])
        result = load_jsonl(tmp_path)
        assert len(result) == 2
        ids = {r["id"] for r in result}
        assert ids == {"a", "b"}

    def test_empty_file(self, tmp_path):
        """空文件应返回空列表。"""
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        result = load_jsonl(f)
        assert result == []


# ============================================================
# stream_jsonl
# ============================================================

class TestStreamJsonl:
    """stream_jsonl 流式迭代测试。"""

    def test_yields_all_records(self, tmp_path):
        """应逐条 yield 所有记录。"""
        records = [{"i": i} for i in range(5)]
        f = tmp_path / "stream.jsonl"
        _write_jsonl(f, records)
        result = list(stream_jsonl(f))
        assert result == records

    def test_is_generator(self, tmp_path):
        """返回值应为生成器（惰性迭代）。"""
        f = tmp_path / "lazy.jsonl"
        _write_jsonl(f, [{"a": 1}])
        gen = stream_jsonl(f)
        # 生成器应有 __next__ 方法
        assert hasattr(gen, "__next__")

    def test_skips_blank_lines(self, tmp_path):
        """流式迭代也应跳过空行。"""
        f = tmp_path / "blanks.jsonl"
        with open(f, "w", encoding="utf-8") as fp:
            fp.write('{"x": 1}\n\n{"x": 2}\n')
        result = list(stream_jsonl(f))
        assert len(result) == 2


# ============================================================
# count_jsonl
# ============================================================

class TestCountJsonl:
    """count_jsonl 行计数测试。"""

    def test_basic_count(self, tmp_path):
        """应正确统计记录数。"""
        f = tmp_path / "count.jsonl"
        _write_jsonl(f, [{"i": i} for i in range(10)])
        assert count_jsonl(f) == 10

    def test_count_skips_blanks(self, tmp_path):
        """空行不计入。"""
        f = tmp_path / "count_blanks.jsonl"
        with open(f, "w", encoding="utf-8") as fp:
            fp.write('{"a": 1}\n\n\n{"b": 2}\n')
        assert count_jsonl(f) == 2

    def test_count_gzip(self, tmp_path):
        """应支持 .jsonl.gz 格式的计数。"""
        f = tmp_path / "count.jsonl.gz"
        _write_gz_jsonl(f, [{"i": i} for i in range(3)])
        assert count_jsonl(f) == 3

    def test_empty_file_count(self, tmp_path):
        """空文件应返回 0。"""
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert count_jsonl(f) == 0

    def test_directory_count(self, tmp_path):
        """目录模式应累加所有 shard 的记录数。"""
        _write_jsonl(tmp_path / "s1.jsonl", [{"a": 1}, {"a": 2}])
        _write_jsonl(tmp_path / "s2.jsonl", [{"a": 3}])
        assert count_jsonl(tmp_path) == 3


# ============================================================
# open_text
# ============================================================

class TestOpenText:
    """open_text 文件句柄测试。"""

    def test_jsonl_file(self, tmp_path):
        """普通 .jsonl 文件应可正常读取。"""
        f = tmp_path / "plain.jsonl"
        f.write_text("hello\n")
        with open_text(f) as fp:
            line = fp.readline()
        assert line.strip() == "hello"

    def test_gzip_file(self, tmp_path):
        """gzip 文件应自动解压读取。"""
        f = tmp_path / "compressed.jsonl.gz"
        _write_gz_jsonl(f, [{"ok": True}])
        with open_text(f) as fp:
            line = fp.readline()
        data = json.loads(line)
        assert data["ok"] is True
