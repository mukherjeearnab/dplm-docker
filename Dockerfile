FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CONDA_ENV_PATH=/opt/conda/envs/dplm
ENV PATH=/opt/conda/envs/dplm/bin:/opt/conda/bin:$PATH

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Accept Anaconda channel Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the Python environment required by DPLM
RUN conda create -y -n dplm python=3.9 pip \
    && conda clean -afy

ENV PATH=/opt/conda/envs/dplm/bin:/opt/conda/bin:$PATH

RUN python --version

WORKDIR /workspace

RUN git clone --recursive https://github.com/bytedance/dplm.git

WORKDIR /workspace/dplm

RUN git checkout 8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d


# Upgrade pip inside the DPLM environment
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.2.0 with CUDA 12.1 support
# RUN python -m pip install \
#     pytorch_lightning==2.2.0 \
#     torchmetrics \
#     torch_geometric \
#     torchtext==0.17.0 \

RUN python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio \
  --index-url https://download.pytorch.org/whl/cpu


RUN python -m pip install --no-build-isolation -e vendor/openfold

# RUN python -m pip install Cython

RUN python -m pip install -e .

# RUN python -m pip uninstall -y torch torchvision torchaudio torch-scatter

# RUN python -m pip install torch torchvision torchaudio \
#         --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

RUN python -m pip install "setuptools<81"

RUN python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

import torch_scatter
print("torch-scatter: OK")

import byprot
print("ByProt: OK")
PY

# Copy the DPLM repository into the image.
# Clone with --recursive before building to include OpenFold.
# COPY . .

# Install DPLM and OpenFold
# RUN sed -i 's/torchtext==0\.17\.0/torchtext/' requirements.txt

# RUN python --version && pip install asdfasdf

# RUN python -m pip install -e .

# RUN python -m pip install --no-build-isolation -e vendor/openfold

# Default working directory
RUN mkdir -p /workspace/dplm/data-bin \
    /workspace/dplm/generation-results

COPY test_gen.py .

CMD ["/bin/bash"]