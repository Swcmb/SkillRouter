"""公共工具模块。

提供模型加载、文本编码、查询/技能格式化、重排提示构造等基础函数，
是 SkillRouter 两阶段检索-重排流水线的核心依赖。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# ---- 共享常量 ----

# 评估数据分层目录映射：tier 名称 -> data_root 下的子目录名
TIER_FILES: dict[str, str] = {
    "easy": "easy",
    "hard": "hard",
}


# ---- 共享工具函数 ----


def aggregate(metrics_list: list[dict]) -> dict:
    """对指标列表取均值，返回聚合结果（含 count 字段）。"""
    if not metrics_list:
        return {}
    out = {}
    for key in metrics_list[0]:
        out[key] = float(np.mean([m[key] for m in metrics_list]))
    out["count"] = len(metrics_list)
    return out


# ---- 评估分层工具 ----


# 三种评估聚合分层：全部 / 单 GT 技能 / 多 GT 技能
_STRATA_KEYS = ("all", "single", "multi")


def make_strata_buckets() -> dict[str, list]:
    """创建空的分层指标收集桶。"""
    return {key: [] for key in _STRATA_KEYS}


def append_to_strata(buckets: dict[str, list], gt_count: int, metrics: dict) -> None:
    """将单条任务的指标追加到对应分层桶中。

    根据 gt_count 将指标追加到 "single"（1 个 GT）或 "multi"（2+ 个 GT），
    同时始终追加到 "all"。

    Args:
        buckets: 由 make_strata_buckets() 创建的桶字典。
        gt_count: 该任务的 ground truth 技能数量。
        metrics: 单条任务的指标字典。
    """
    buckets["all"].append(metrics)
    buckets["single" if gt_count == 1 else "multi"].append(metrics)


def filter_tasks_by_mode(
    tasks: list[dict],
    relevance: dict,
    task_mode: str,
) -> list[dict]:
    """根据 task_mode 过滤任务列表。

    Args:
        tasks: 全部任务列表。
        relevance: relevance.json 内容，{task_id: {...}}。
        task_mode: "core" 排除 generic_only；"single" 仅保留单 GT 技能任务。

    Returns:
        过滤后的任务列表。
    """
    if task_mode == "all":
        return tasks
    filtered = []
    for task in tasks:
        rel_entry = relevance.get(task["task_id"], {})
        task_type = rel_entry.get("task_type")
        # core 模式：排除 generic_only 类型任务
        if task_mode == "core" and task_type == "generic_only":
            continue
        # single 模式：仅保留恰好有 1 个 GT 技能的任务
        if task_mode == "single":
            gt_ids = set(rel_entry.get("gt_skill_ids", []))
            if len(gt_ids) != 1:
                continue
        filtered.append(task)
    return filtered


QUERY_INSTRUCTION = (
    "Instruct: Given a coding task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)
"""检索阶段使用的查询指令前缀，用于构造 in-context learning 格式的查询。"""

RERANK_INSTRUCTION = (
    "Given a coding task description, judge whether the skill document "
    "is relevant and useful for completing the task"
)
"""重排阶段使用的判断指令，提示模型评估技能文档与查询的相关性。"""


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则递归创建。

    Args:
        path: 目录路径，可以是字符串或 Path 对象。

    Returns:
        创建或已存在的 Path 对象。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> torch.device:
    """获取可用的计算设备。

    优先使用 CUDA GPU，不可用时回退到 CPU。

    Returns:
        torch.device 实例（"cuda" 或 "cpu"）。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """从因果语言模型的最后一个隐藏状态中提取序列表示。

    对于左填充（left-padding）模型取最后一个 token 的隐藏状态；
    对于右填充模型，取每个序列实际最后一个有效 token 的隐藏状态。
    这是 Embedding 模型池化策略的核心实现。

    Args:
        last_hidden_states: 模型最后一层隐藏状态，形状 (batch, seq_len, hidden_dim)。
        attention_mask: 注意力掩码，形状 (batch, seq_len)。

    Returns:
        池化后的序列表示，形状 (batch, hidden_dim)。
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def format_query(raw_query: str, max_len: int = 1500) -> str:
    """将原始任务查询格式化为带指令前缀的检索查询。

    在查询文本前拼接 QUERY_INSTRUCTION 指令前缀，并截断到最大长度。

    Args:
        raw_query: 原始任务描述文本。
        max_len: 查询文本的最大字符数，默认 1500。

    Returns:
        格式化后的查询字符串，格式为 "Instruct:...Query:<query_text>"。
    """
    return f"{QUERY_INSTRUCTION}{raw_query[:max_len]}"


def format_skill(skill: dict, desc_max: int = 300, body_max: int = 2500) -> str:
    """将技能字典格式化为 "名称 | 描述 | 正文" 的平面文本。

    用于 Stage 1 检索阶段，将技能的完整信息拼接为单个字符串
    供 Embedding 模型编码。

    Args:
        skill: 技能字典，包含 name、description、body 等字段。
        desc_max: 描述文本的最大截断长度，默认 300。
        body_max: 正文的最大截断长度，默认 2500。

    Returns:
        格式化后的技能文本，格式为 "<name> | <description> | <body>"。
    """
    name = skill.get("name", "")
    desc = (skill.get("description") or "")[:desc_max]
    body = (skill.get("body") or "")[:body_max]
    return f"{name} | {desc} | {body}"


def format_rerank_prompt(
    name: str,
    desc: str,
    body: str,
    query_text: str,
    prompt_format: str = "flat-full",
    desc_max: int = 500,
    body_max: int = 2000,
) -> str:
    """构造重排模型的输入提示。

    支持三种提示格式：flat-full（名称|描述|正文）、flat-nd（名称|描述）、
    struct（XML 标签分隔），用于 Stage 2 重排阶段。

    Args:
        name: 技能名称。
        desc: 技能描述文本。
        body: 技能正文文本。
        query_text: 任务查询文本。
        prompt_format: 提示格式，可选 "flat-full"、"flat-nd"、"struct"。
        desc_max: 描述文本的最大截断长度，默认 500。
        body_max: 正文的最大截断长度，默认 2000。

    Returns:
        完整的重排提示字符串，包含指令、查询和技能文档。

    Raises:
        ValueError: 当 prompt_format 不是已知格式时抛出。
    """
    desc = desc[:desc_max]
    body = body[:body_max]

    if prompt_format == "flat-nd":
        doc_text = f"{name} | {desc}"
    elif prompt_format == "flat-full":
        doc_text = f"{name} | {desc} | {body}"
    elif prompt_format == "struct":
        return (
            f"<Instruct>: {RERANK_INSTRUCTION}\n\n"
            f"<Query>: {query_text}\n\n"
            f"<Skill>:\n"
            f"<Name>: {name}\n"
            f"<Description>: {desc}\n"
            f"<Body>: {body}"
        )
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}")

    return (
        f"<Instruct>: {RERANK_INSTRUCTION}\n\n"
        f"<Query>: {query_text}\n\n"
        f"<Document>: {doc_text}"
    )


def _ensure_pad_token(tokenizer) -> None:
    """确保分词器具有 pad_token，缺失时回退到 eos_token。"""
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def load_embedding_model(model_name_or_path: str, dtype: torch.dtype = torch.bfloat16):
    """加载检索阶段的 Embedding 模型和分词器。

    使用 AutoModel 加载预训练 Embedding 模型，设置左填充以适配
    last_token_pool 池化策略。若缺少 pad_token 则自动回退到 eos_token。

    Args:
        model_name_or_path: HuggingFace 模型名称或本地路径。
        dtype: 模型权重的数据类型，默认 bfloat16。

    Returns:
        (model, tokenizer) 元组。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="left",
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    _ensure_pad_token(tokenizer)
    return model, tokenizer


def load_reranker_model(model_name_or_path: str, dtype: torch.dtype = torch.bfloat16):
    """加载重排阶段的因果语言模型和分词器。

    使用 AutoModelForCausalLM 加载重排模型，通过 yes/no logit 差值
    计算相关性分数。设置左填充，缺少 pad_token 时回退到 eos_token。

    Args:
        model_name_or_path: HuggingFace 模型名称或本地路径。
        dtype: 模型权重的数据类型，默认 bfloat16。

    Returns:
        (model, tokenizer) 元组。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    _ensure_pad_token(tokenizer)
    return model, tokenizer


def encode_texts(model, tokenizer, texts: list[str], max_length: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """批量编码文本列表为 L2 归一化的嵌入向量。

    使用 last_token_pool 池化策略从因果 LM 的隐藏状态中提取表示，
    并进行 L2 归一化以便直接计算余弦相似度。自动处理 GPU/CPU 切换。

    Args:
        model: 已加载的 Embedding 模型。
        tokenizer: 对应的分词器。
        texts: 待编码的文本列表。
        max_length: 分词的最大 token 数。
        batch_size: 批处理大小。
        device: 计算设备。

    Returns:
        形状 (len(texts), hidden_dim) 的嵌入张量，已在 CPU 上。
    """
    model.eval()
    all_embs: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            embs = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


def get_reranker_template_tokens(tokenizer):
    """获取重排模型 ChatML 模板的前缀和后缀 token 序列。

    重排模型使用固定的系统消息（"Judge whether..."）作为前缀，
    以及 "" 后缀。这些 token 被预编码以便在批量推理中复用。

    Args:
        tokenizer: 重排模型对应的分词器。

    Returns:
        (prefix_tokens, suffix_tokens) 元组，各为 token ID 列表。
    """
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements '
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    return prefix_tokens, suffix_tokens


def tokenize_reranker_text(text: str, tokenizer, prefix_tokens, suffix_tokens, max_length: int) -> list[int]:
    """将重排文本与模板前缀/后缀拼接后进行分词。

    将用户侧文本（指令+查询+文档）截断到合理长度后，
    与预编码的 ChatML 系统消息前缀和后缀拼接为完整的 token 序列。

    Args:
        text: 用户侧文本（指令+查询+文档）。
        tokenizer: 分词器。
        prefix_tokens: ChatML 系统消息的 token ID 列表。
        suffix_tokens: ChatML 后缀的 token ID 列表。
        max_length: 完整序列的最大 token 数（含前缀和后缀）。

    Returns:
        拼接后的完整 token ID 列表。
    """
    inputs = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        return_attention_mask=False,
    )
    return prefix_tokens + inputs["input_ids"] + suffix_tokens


def score_candidates_with_reranker(
    model,
    tokenizer,
    query_text: str,
    candidates: list[dict],
    prompt_format: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    """使用重排模型对候选技能进行批量相关性评分。

    对每个 (query, candidate) 对计算 P("yes") - P("no") logit 差值
    作为相关性分数，支持批量推理和左填充。

    Args:
        model: 已加载的因果语言重排模型。
        tokenizer: 对应的分词器。
        query_text: 任务查询文本。
        candidates: 候选技能字典列表，每个需包含 name/body 等字段。
        prompt_format: 提示格式（"flat-full"、"flat-nd"、"struct"）。
        max_length: 完整序列的最大 token 数。
        batch_size: 推理批处理大小。
        device: 计算设备。

    Returns:
        与 candidates 等长的相关性分数列表。
    """
    prefix_tokens, suffix_tokens = get_reranker_template_tokens(tokenizer)
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")

    # 构造重排提示文本
    texts = [
        format_rerank_prompt(
            cand["name"],
            cand.get("description", cand.get("desc", "")),
            cand["body"],
            query_text,
            prompt_format=prompt_format,
        )
        for cand in candidates
    ]
    tokenized = [
        tokenize_reranker_text(text, tokenizer, prefix_tokens, suffix_tokens, max_length)
        for text in texts
    ]

    scores: list[float] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for i in range(0, len(tokenized), batch_size):
        batch_ids = tokenized[i:i + batch_size]
        # 左填充：将变长序列对齐到批次内最大长度（因果 LM 需要左填充以正确获取末尾 logits）
        max_len = max(len(ids) for ids in batch_ids)
        input_ids_list, mask_list = [], []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            input_ids_list.append([pad_id] * pad_len + ids)
            mask_list.append([0] * pad_len + [1] * len(ids))
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        attention_mask = torch.tensor(mask_list, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
            batch_scores = (logits[:, token_true_id] - logits[:, token_false_id]).float().cpu().tolist()
        scores.extend(batch_scores)
    return scores

