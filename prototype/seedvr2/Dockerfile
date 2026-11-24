FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

ARG NUMZ_REPO_REF=4490bd1f482e026674543386bb2a4d176da245b9
ARG EXPECTED_TORCH_VERSION=2.14.1
ARG EXPECTED_TORCHVISION_VERSION=0.29.1
ARG EXPECTED_CUDA_VERSION=13.0
ARG VENV_PATH=/opt/seedvr2-venv

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=${VENV_PATH} \
    PATH=${VENV_PATH}/bin:${PATH}

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg git ca-certificates python3-venv; \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /opt/seedvr2 \
    && cd /opt/seedvr2 \
    && git checkout --detach "${NUMZ_REPO_REF}"

COPY prototype/seedvr2/requirements.constraints.txt /tmp/requirements.constraints.txt
RUN python -m venv --system-site-packages "${VENV_PATH}" \
    && "${VENV_PATH}/bin/python" -m pip install --upgrade pip \
    && grep -Ev '^(torch|torchvision)([[:space:]]|[<>=!~]).*$|^(torch|torchvision)$' /opt/seedvr2/requirements.txt > /tmp/requirements.seedvr2.txt \
    && "${VENV_PATH}/bin/python" -m pip install --prefer-binary -c /tmp/requirements.constraints.txt -r /tmp/requirements.seedvr2.txt \
    && "${VENV_PATH}/bin/python" -m pip install pydantic==2.9.2 \
    && "${VENV_PATH}/bin/python" -c "import os, sys, torch, torchvision; assert sys.executable.startswith(os.environ['VENV_PATH']), sys.executable; assert torch.__version__.startswith('${EXPECTED_TORCH_VERSION}'), torch.__version__; assert torchvision.__version__.startswith('${EXPECTED_TORCHVISION_VERSION}'), torchvision.__version__; assert str(torch.version.cuda).startswith('${EXPECTED_CUDA_VERSION}'), torch.version.cuda" \
    && "${VENV_PATH}/bin/python" /opt/seedvr2/inference_cli.py --help > /tmp/seedvr2-help.txt \
    && grep -q -- '--dit_model' /tmp/seedvr2-help.txt \
    && grep -q -- '--attention_mode' /tmp/seedvr2-help.txt \
    && grep -q -- '--blocks_to_swap' /tmp/seedvr2-help.txt \
    && grep -q -- '--vae_encode_tiled' /tmp/seedvr2-help.txt \
    && rm -f /tmp/seedvr2-help.txt /tmp/requirements.seedvr2.txt

RUN mkdir -p /opt/prototype/src/vast_upscaler /workspace/input /workspace/output /workspace/cache/seedvr2 /workspace/tmp
COPY src/vast_upscaler/seedvr2_prototype.py /opt/prototype/src/vast_upscaler/seedvr2_prototype.py
COPY src/vast_upscaler/video.py /opt/prototype/src/vast_upscaler/video.py
COPY scripts/seedvr2_prototype.py /opt/prototype/seedvr2_prototype.py

ENV PYTHONPATH=/opt/prototype/src

WORKDIR /workspace
ENTRYPOINT ["python", "/opt/prototype/seedvr2_prototype.py"]

