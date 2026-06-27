# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SkillRouter is a retrieval system for routing user tasks to the correct skill in large-scale LLM Agent skill registries (~80K skills). The key finding: when skills are metadata-similar at scale, the full skill body (not just name/description) is the decisive routing signal. It uses a two-stage pipeline: 0.6B embedding encoder for retrieval + 0.6B causal LM reranker.

Paper: arXiv:2603.22455 (2026)

## Setup & Common Commands

```bash
# Install dependencies (use conda for CUDA/torch per global rules)
pip install -r requirements.txt

# Download benchmark data (~80K skills, 87 tasks)
bash scripts/download_eval_data.sh
# or: make download-data

# Run full evaluation with published models
bash scripts/evaluate_open_models.sh
# or: make eval-open-models

# Evaluate custom model (retrieval only)
python3 -m src.export_retrieval \
  --encoder_model_or_path /path/to/encoder \
  --data_root data/eval_core \
  --output_dir outputs/custom_eval \
  --tiers easy hard

# Evaluate custom model (retrieval + reranking)
python3 -m src.run_open_model_eval \
  --data_root data/eval_core \
  --encoder_model_or_path /path/to/encoder \
  --reranker_model_or_path /path/to/reranker \
  --tiers easy hard \
  --output_dir outputs/custom_pipeline_eval

# Score existing predictions
bash scripts/evaluate_predictions.sh \
  --predictions outputs/custom_eval/retrieval/easy.json \
  --tier easy
```

**Note**: `scripts/download_eval_data.sh` requires `huggingface_hub` (not in requirements.txt).

## Environment Variables

| Variable | Purpose |
|:---|:---|
| `SKILLROUTER_EMB_MODEL_OR_PATH` | Override default embedding model |
| `SKILLROUTER_RERANK_MODEL_OR_PATH` | Override default reranker model |
| `SKILLROUTER_DATA_REPO` | Override HuggingFace data repo |
| `SKILLROUTER_DATA_DIR` | Override local data directory |

## Architecture: Two-Stage Retrieval-Reranking Pipeline

```
Task Query → format_query() → [Embedding-0.6B] → cosine similarity → Top-K → [Reranker-0.6B] → ranked results
```

**Stage 1 — Embedding Retrieval**: Encodes query and all ~80K skills via `last_token_pool` (causal LM hidden states), computes cosine similarity matrix `Q @ P^T`, retrieves top-20 candidates.

**Stage 2 — Reranking**: For each (query, candidate) pair, uses chat template to compute `P("yes") - P("no")` logit difference as relevance score. The reranker wraps each prompt in a fixed ChatML system message (see `get_reranker_template_tokens` in `common.py`). Supports three prompt formats:

| Format | Name | Skill content included |
|:---|:---|:---|
| `flat-full` | name\|desc\|body | Full skill text (default) |
| `flat-nd` | name\|desc | Metadata only (no body) |
| `struct` | XML-tagged | Name/description/body in separate XML tags |

## Code Layering

```
Layer 0 (Foundation):
  data_io.py      — JSONL/gzip data loading, no external deps
  metrics.py      — 12 IR evaluation metrics (numpy only)

Layer 1 (Utilities):
  common.py       — Model loading, tokenization, encoding, text formatting (torch + transformers)

Layer 2 (CLI Entrypoints):
  run_open_model_eval.py  — End-to-end eval (retrieval + rerank + metrics)
  export_retrieval.py     — Export top-K retrieval results only
  evaluate_predictions.py — Score existing prediction files
```

Zero circular dependencies. Each entrypoint composes the foundation modules it needs.

## Key Abstractions

- **`last_token_pool`** (`common.py`): Extracts sequence representation from causal LM's last hidden state, handles left/right padding automatically
- **`encode_texts`** (`common.py`): Batch encoding with L2 normalization, GPU/CPU adaptive
- **`score_candidates_with_reranker`** (`run_open_model_eval.py`): Reranking core using yes/no logit difference
- **`compute_all_metrics`** (`metrics.py`): Unified metric computation supporting binary and graded relevance
- **`FullCoverage`** (`metrics.py`): Custom metric measuring whether top-K covers all required skills

## Published Models

| Model | Role | HuggingFace |
|:---|:---|:---|
| SkillRouter-Embedding-0.6B | Stage 1 retrieval | `pipizhao/SkillRouter-Embedding-0.6B` |
| SkillRouter-Reranker-0.6B | Stage 2 reranking | `pipizhao/SkillRouter-Reranker-0.6B` |

## Evaluation Data & Scoring Protocol

- **Tasks**: 87 benchmark tasks in `data/eval_core/tasks.jsonl` (75 participate in scoring after excluding `generic_only`)
- **Skill pools**: `easy/` (~78K) and `hard/` (~79K) tiers, 10 `.jsonl.gz` shards each
- **Ground truth**: `data/eval_core/relevance.json` with binary and graded relevance labels
- **Metrics**: nDCG@{1,3,10}, Hit@1, Precision@3, MRR@10, Recall@{10,20,50}, FullCoverage@{3,5,10}
- **Prediction format**: JSON object mapping `task_id → [skill_id, ...]` (see `evaluation/example_retrieval_submission.json`)
- **Aggregation strata**: results are split into `all`, `single` (1 GT skill), `multi` (2+ GT skills) subgroups
- **`task_mode`**: `core` (default, excludes `generic_only`), `all` (all tasks), `single` (single-GT-skill tasks only)
- **Retrieval truncation**: `format_skill` uses `desc_max=300, body_max=2500` by default; `run_open_model_eval` overrides to `desc_max=500, body_max=8000` during full pipeline eval
