from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common import (
    TIER_FILES,
    aggregate,
    append_to_strata,
    encode_texts,
    ensure_dir,
    filter_tasks_by_mode,
    format_query,
    format_skill,
    get_device,
    load_embedding_model,
    load_reranker_model,
    make_strata_buckets,
    score_candidates_with_reranker,
)
from src.data_io import load_jsonl
from src.metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="One-click local evaluation for the open SkillRouter 0.6B models.")
    parser.add_argument("--data_root", default="data/eval_core")
    parser.add_argument("--encoder_model_or_path", default="pipizhao/SkillRouter-Embedding-0.6B")
    parser.add_argument("--reranker_model_or_path", default="pipizhao/SkillRouter-Reranker-0.6B")
    parser.add_argument("--task_mode", choices=["core", "all", "single"], default="core")
    parser.add_argument("--tiers", nargs="+", choices=["easy", "hard"], default=["easy", "hard"])
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--encoder_max_length", type=int, default=4096)
    parser.add_argument("--reranker_max_length", type=int, default=4096)
    parser.add_argument("--encoder_batch_size", type=int, default=32)
    parser.add_argument("--reranker_batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default="outputs/open_model_eval")
    parser.add_argument("--prompt_format", default="flat-full", choices=["flat-full", "flat-nd", "struct"])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = ensure_dir(args.output_dir)
    retrieval_dir = ensure_dir(output_dir / "retrieval")
    reranked_dir = ensure_dir(output_dir / "reranked")

    device = get_device()
    try:
        emb_model, emb_tokenizer = load_embedding_model(args.encoder_model_or_path)
    except Exception as exc:
        raise SystemExit(
            "Failed to load the embedding model. "
            "Pass a valid local checkpoint path or a Hugging Face repo ID "
            "that contains model weights."
        ) from exc
    emb_model.to(device).eval()
    try:
        rr_model, rr_tokenizer = load_reranker_model(args.reranker_model_or_path)
    except Exception as exc:
        raise SystemExit(
            "Failed to load the reranker model. "
            "Pass a valid local checkpoint path or a Hugging Face repo ID "
            "that contains model weights."
        ) from exc
    rr_model.to(device).eval()

    tasks = load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text())

    filtered_tasks = filter_tasks_by_mode(tasks, relevance, args.task_mode)

    query_texts = [format_query(t["instruction_text"], max_len=2000) for t in filtered_tasks]
    query_embs = encode_texts(
        emb_model, emb_tokenizer, query_texts, args.encoder_max_length, args.encoder_batch_size, device
    )

    summary = {}
    for tier in args.tiers:
        tier_stem = TIER_FILES[tier]
        pool = load_jsonl(data_root / tier_stem)
        pool_ids = [skill["skill_id"] for skill in pool]
        pool_id_set = set(pool_ids)
        pool_texts = [format_skill(skill, desc_max=500, body_max=8000) for skill in pool]
        pool_embs = encode_texts(
            emb_model, emb_tokenizer, pool_texts, args.encoder_max_length, args.encoder_batch_size, device
        )
        sim_matrix = query_embs @ pool_embs.T

        retrieval_results = {}
        reranked_results = {}
        metrics_retrieval = make_strata_buckets()
        metrics_pipeline = make_strata_buckets()

        for qi, task in enumerate(filtered_tasks):
            task_id = task["task_id"]
            rel_entry = relevance[task_id]
            if args.task_mode == "core":
                gt_ids = set(rel_entry.get("core_gt_ids", rel_entry.get("gt_skill_ids", [])))
            else:
                gt_ids = set(rel_entry.get("gt_skill_ids", []))
            gt_ids = gt_ids & pool_id_set
            if not gt_ids:
                continue

            tier_relevance = {k: float(v) for k, v in rel_entry.get("relevance", {}).items() if k in pool_id_set}
            sims = sim_matrix[qi]
            _, topk_idx = torch.topk(sims, min(args.retrieval_top_k, len(pool_ids)))
            top_ids = [pool_ids[idx] for idx in topk_idx.tolist()]
            retrieval_results[task_id] = top_ids
            m_ret = compute_all_metrics(top_ids, gt_ids, tier_relevance or None)
            append_to_strata(metrics_retrieval, len(gt_ids), m_ret)

            candidates = [pool[idx] for idx in topk_idx.tolist()]
            scores = score_candidates_with_reranker(
                rr_model,
                rr_tokenizer,
                task["instruction_text"],
                candidates,
                args.prompt_format,
                args.reranker_max_length,
                args.reranker_batch_size,
                device,
            )
            # 按重排分数降序排列
            ranked_pairs = sorted(zip(top_ids, scores), key=lambda pair: pair[1], reverse=True)
            reranked_ids = [sid for sid, _ in ranked_pairs]
            reranked_results[task_id] = reranked_ids
            m_pipe = compute_all_metrics(reranked_ids, gt_ids, tier_relevance or None)
            append_to_strata(metrics_pipeline, len(gt_ids), m_pipe)

        (retrieval_dir / f"{tier}.json").write_text(json.dumps(retrieval_results, indent=2, ensure_ascii=False))
        (reranked_dir / f"{tier}.json").write_text(json.dumps(reranked_results, indent=2, ensure_ascii=False))
        summary[tier] = {
            "retrieval": {k: aggregate(v) for k, v in metrics_retrieval.items() if v},
            "pipeline": {k: aggregate(v) for k, v in metrics_pipeline.items() if v},
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
