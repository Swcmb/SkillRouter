# SkillRouter Docker 镜像
# 支持 CPU 和 CUDA 两种模式

# ---- 基础镜像选择 ----
# 默认使用 CPU 版本；需要 GPU 时改用 nvidia/cuda 镜像
ARG BASE_IMAGE=python:3.12-slim

FROM ${BASE_IMAGE}

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# 使用 --no-cache-dir 减少镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 安装项目本身（可编辑模式）
RUN pip install --no-cache-dir -e .

# 创建数据和输出目录
RUN mkdir -p /app/data /app/outputs

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 默认入口：运行测试
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
