FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

ARG NUMZ_REPO_REF=4490bd1f482e026674543386bb2a4d176da245b9
ARG EXPECTED_TORCH_VERSION=2.14.1
ARG EXPECTED_TORCHVISION_VERSION=0.29.1
ARG EXPECTED_CUDA_VERSION=13.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg git ca-certificates; \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /opt/seedvr2 \
    && cd /opt/seedvr2 \
    && git checkout --detach "${NUMZ_REPO_REF}"

COPY prototype/seedvr2/requirements.constraints.txt /tmp/requirements.constraints.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --prefer-binary -c /tmp/requirements.constraints.txt -r /opt/seedvr2/requirements.txt \
    && python -c "import torch, torchvision; assert torch.__version__.startswith('${EXPECTED_TORCH_VERSION}'), torch.__version__; assert torchvision.__version__.startswith('${EXPECTED_TORCHVISION_VERSION}'), torchvision.__version__; assert str(torch.version.cuda).startswith('${EXPECTED_CUDA_VERSION}'), torch.version.cuda"

RUN mkdir -p /opt/prototype/src/vast_upscaler /workspace/input /workspace/output /workspace/cache/seedvr2 /workspace/tmp
COPY src/vast_upscaler/seedvr2_prototype.py /opt/prototype/src/vast_upscaler/seedvr2_prototype.py
COPY scripts/seedvr2_prototype.py /opt/prototype/seedvr2_prototype.py

ENV PYTHONPATH=/opt/prototype/src

WORKDIR /workspace
ENTRYPOINT ["python", "/opt/prototype/seedvr2_prototype.py"]

