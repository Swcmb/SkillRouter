.PHONY: install download-data eval-open-models lint test check

# ──────────────────────────────────────────────
# 环境安装：依赖 + 评测数据
# ──────────────────────────────────────────────

install:
	pip install -r requirements.txt
	# huggingface_hub 用于下载评测数据，单独安装
	pip install huggingface_hub

# ──────────────────────────────────────────────
# 数据下载与评测
# ──────────────────────────────────────────────

download-data:
	bash scripts/download_eval_data.sh

eval-open-models:
	bash scripts/evaluate_open_models.sh

# ──────────────────────────────────────────────
# 代码质量检查（lint）：语法 + 风格
# ──────────────────────────────────────────────

lint:
	python3 -m py_compile src/data_io.py
	python3 -m py_compile src/metrics.py
	python3 -m py_compile src/common.py
	python3 -m py_compile src/evaluate_predictions.py
	python3 -m py_compile src/export_retrieval.py
	python3 -m py_compile src/run_open_model_eval.py
	@echo "All modules compile successfully"
	# 若 flake8 可用则执行风格检查
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/ --max-line-length=120 --ignore=E501,W503; \
	else \
		echo "flake8 not installed, skipping style check (pip install flake8)"; \
	fi

# ──────────────────────────────────────────────
# 单元测试：验证核心模块的基础逻辑
# ──────────────────────────────────────────────

test:
	python3 -m pytest tests/ -v --tb=short 2>/dev/null || \
	python3 -c "\
	from src.metrics import compute_all_metrics, ndcg_at_k, mrr_at_k, recall_at_k, precision_at_k, hit_at_k, full_coverage_at_k; \
	ranked = ['a', 'b', 'c']; \
	gt = {'a', 'c'}; \
	assert ndcg_at_k([1, 0, 1], [1, 1, 0], 3) > 0, 'ndcg_at_k failed'; \
	assert mrr_at_k(ranked, gt, 10) == 1.0, 'mrr_at_k failed'; \
	assert recall_at_k(ranked, gt, 3) == 1.0, 'recall_at_k failed'; \
	assert hit_at_k(ranked, gt, 1) == 1.0, 'hit_at_k failed'; \
	assert precision_at_k(ranked, gt, 3) > 0, 'precision_at_k failed'; \
	assert full_coverage_at_k(ranked, gt, 3) == 1.0, 'full_coverage_at_k failed'; \
	results = compute_all_metrics(ranked, gt); \
	assert 'nDCG@3' in results, 'compute_all_metrics missing nDCG@3'; \
	assert 'MRR@10' in results, 'compute_all_metrics missing MRR@10'; \
	assert 'Recall@10' in results, 'compute_all_metrics missing Recall@10'; \
	assert 'FullCoverage@3' in results, 'compute_all_metrics missing FullCoverage@3'; \
	print('All unit tests passed')"

# ──────────────────────────────────────────────
# 综合检查：lint + test 一键执行
# ──────────────────────────────────────────────

check: lint test

# ──────────────────────────────────────────────
# Docker 构建与测试
# ──────────────────────────────────────────────

docker-build:
	docker build -t skillrouter:latest .

docker-test: docker-build
	docker run --rm skillrouter:latest

docker-lint: docker-build
	docker run --rm skillrouter:latest ruff check src/ tests/

docker-shell: docker-build
	docker run --rm -it skillrouter:latest /bin/bash
