"""数据 I/O 模块。

提供 JSONL / gzip JSONL 文件的发现、读取、流式加载和计数功能，
是 SkillRouter 项目中所有数据加载的基础组件，无外部依赖。
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path

# JSONL 文件后缀集合，用于文件发现与过滤
_JSONL_SUFFIXES = (".jsonl", ".jsonl.gz")


def _is_jsonl(path: Path) -> bool:
    """判断文件路径是否为 JSONL 格式（含 gzip 压缩）。"""
    return any(path.name.endswith(sfx) for sfx in _JSONL_SUFFIXES)


def iter_jsonl_paths(path: str | Path) -> list[Path]:
    """发现指定路径下的所有 JSONL 文件（含 gzip 压缩）。

    如果 path 是单个文件且后缀匹配则直接返回；如果 path 是目录
    则遍历其直接子文件并按名称排序返回。不递归搜索子目录。

    Args:
        path: 文件路径或目录路径。

    Returns:
        排序后的 .jsonl / .jsonl.gz 文件路径列表。

    Raises:
        FileNotFoundError: 当路径不存在或既不是文件也不是目录时。
    """
    p = Path(path)
    if p.is_file() and _is_jsonl(p):
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.is_file() and _is_jsonl(f))
    raise FileNotFoundError(f"Path not found: {p}")


def open_text(path: str | Path):
    """以文本模式打开文件，自动处理 gzip 压缩。

    根据文件后缀判断是否为 gzip 格式，.gz 文件使用 gzip.open，
    普通文件使用内置 open，统一使用 UTF-8 编码。

    Args:
        path: 文件路径。

    Returns:
        可迭代的文本文件对象。
    """
    p = Path(path)
    if p.name.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return p.open("r", encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件的全部记录到内存列表。

    支持单文件或目录模式，自动跳过空行。适用于数据量较小的场景
    （如评估任务列表、ground truth 等）。大数据集建议使用 stream_jsonl。

    Args:
        path: JSONL 文件路径或包含 JSONL 文件的目录路径。

    Returns:
        所有 JSON 记录的列表，每条记录为一个字典。
    """
    records: list[dict] = []
    for file_path in iter_jsonl_paths(path):
        with open_text(file_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def stream_jsonl(path: str | Path) -> Iterable[dict]:
    """流式逐条读取 JSONL 文件记录（惰性生成器）。

    与 load_jsonl 不同，此函数以生成器方式逐条产出记录，
    不会一次性将全部数据加载到内存，适用于大规模数据集（如 ~80K 技能池）。

    Args:
        path: JSONL 文件路径或包含 JSONL 文件的目录路径。

    Yields:
        逐条产出的 JSON 字典记录。
    """
    for file_path in iter_jsonl_paths(path):
        with open_text(file_path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def count_jsonl(path: str | Path) -> int:
    """统计 JSONL 文件中的有效记录总数。

    仅计数非空行，不解析 JSON 内容，效率高于 list(load_jsonl(...))。
    支持单文件或目录模式。

    Args:
        path: JSONL 文件路径或包含 JSONL 文件的目录路径。

    Returns:
        有效记录的数量。
    """
    count = 0
    for file_path in iter_jsonl_paths(path):
        with open_text(file_path) as f:
            for line in f:
                if line.strip():
                    count += 1
    return count
