FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04


# 非対話モード＆タイムゾーン設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt-get update && \
    apt-get install -y tzdata \
    wget \
    git \
    curl \
    unzip \
    nano \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# Minicondaインストール
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"


# GPUライブラリのパス追加
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
RUN ln -sf /usr/local/cuda-11.8/lib64/libcusolver.so.11 /usr/local/cuda-11.8/lib64/libcusolver.so.10


# Conda仮想環境の作成と必要ライブラリのインストール
RUN conda create -n umamba python=3.10 -y

# umamba環境を有効化して必要なライブラリをインストール
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate umamba && \
    pip install --upgrade pip setuptools wheel"

# PyTorch 2.0.1をインストール
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate umamba && \
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install transformers==4.27.4"

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate umamba && \
    pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post1/causal_conv1d-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install https://github.com/state-spaces/mamba/releases/download/v1.0.1/mamba_ssm-1.0.1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# キャッシュ削除で軽量化
RUN conda clean -afy

