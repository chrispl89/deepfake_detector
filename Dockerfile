# Dockerfile for deepfake detection system
# Supports both CPU and GPU (CUDA) execution

ARG CUDA_VERSION=11.8.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS gpu-base

# CPU-only base image
FROM ubuntu:22.04 AS cpu-base

# Choose base based on build argument
ARG USE_GPU=false
FROM ${USE_GPU:+gpu-base}${USE_GPU:-cpu-base} AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-minimal.txt ./

# Install Python dependencies
ARG INSTALL_MODE=minimal
RUN if [ "$INSTALL_MODE" = "full" ]; then \
        pip3 install --no-cache-dir -r requirements.txt; \
    else \
        pip3 install --no-cache-dir -r requirements-minimal.txt; \
    fi

# Install PyTorch with appropriate backend
ARG USE_GPU=false
RUN if [ "$USE_GPU" = "true" ]; then \
        pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/datasets checkpoints results logs

# Set permissions
RUN chmod +x inference.py train.py

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python3", "inference.py", "--help"]
