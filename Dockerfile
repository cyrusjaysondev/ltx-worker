FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:/root/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# Install uv for fast package installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone LTX-2 and install pipelines
RUN git clone https://github.com/Lightricks/LTX-2 /tmp/ltx2
RUN uv pip install --system /tmp/ltx2/packages/ltx-core
RUN uv pip install --system /tmp/ltx2/packages/ltx-pipelines

# Install PyTorch with CUDA 12.4
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall

# Install flash-attn prebuilt wheel (cu124 + torch2.4 + python3.10)
RUN pip install \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/flash_attn-2.6.3+cu124torch2.4-cp310-cp310-linux_x86_64.whl

# Install worker dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
