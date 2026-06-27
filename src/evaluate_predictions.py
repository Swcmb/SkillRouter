from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import TIER_FILES, aggregate, append_to_strata, filter_tasks_by_mode, make_strata_buckets
from src.data_io import load_jsonl
from src.metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval or reranked predictions on SkillRouter Eval Core.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--task_mode", choices=["core", "all", "single"], default="core")
    parser.add_argument("--tier", choices=["easy", "hard"], required=True)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    tier_pool = load_jsonl(data_root / TIER_FILES[args.tier])
    pool_id_set = {skill["skill_id"] for skill in tier_pool}
    tasks = load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text())
    predictions = json.loads(Path(args.predictions).read_text())

    filtered_tasks = filter_tasks_by_mode(tasks, relevance, args.task_mode)
    results_by_stratum = make_strata_buckets()

    for task in filtered_tasks:
        task_id = task["task_id"]
        rel_entry = relevance.get(task_id, {})
        # core 模式使用 core_gt_ids，其余使用 gt_skill_ids
        if args.task_mode == "core":
            gt_ids = set(rel_entry.get("core_gt_ids", rel_entry.get("gt_skill_ids", [])))
        else:
            gt_ids = set(rel_entry.get("gt_skill_ids", []))

        gt_ids_in_pool = gt_ids & pool_id_set
        if not gt_ids_in_pool or task_id not in predictions:
            continue

        ranked_ids = predictions[task_id]
        tier_relevance = {
            k: float(v) for k, v in rel_entry.get("relevance", {}).items() if k in pool_id_set
        }
        metrics = compute_all_metrics(ranked_ids, gt_ids_in_pool, tier_relevance or None)
        append_to_strata(results_by_stratum, len(gt_ids), metrics)

    aggregated = {k: aggregate(v) for k, v in results_by_stratum.items() if v}
    text = json.dumps(aggregated, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text)


if __name__ == "__main__":
    main()
