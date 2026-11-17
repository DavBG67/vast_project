FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ARG REALESRGAN_X4PLUS_URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
ARG REALESRGAN_X4PLUS_SHA256=4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKER_HOME=/app/worker \
    MODELS_DIR=/opt/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${WORKER_HOME}

COPY worker/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    # Install pinned runtime deps except realesrgan/basicsr first.
    # This keeps torch/torchvision from the CUDA12.1 base image as the active runtime.
    && grep -v -E '^(realesrgan|basicsr)==' requirements.txt > requirements.core.txt \
    && pip install --prefer-binary -r requirements.core.txt \
    # Install basicsr explicitly without build isolation so setup.py reuses already-installed torch.
    && pip install --no-build-isolation basicsr==1.4.2 \
    # Install realesrgan without dependency resolution to avoid re-triggering basicsr install.
    && pip install --no-deps realesrgan==0.3.0 \
    && rm -f requirements.core.txt

RUN mkdir -p ${MODELS_DIR} \
    && curl -L -o ${MODELS_DIR}/RealESRGAN_x4plus.pth \
            ${REALESRGAN_X4PLUS_URL} \
        && echo "${REALESRGAN_X4PLUS_SHA256}  ${MODELS_DIR}/RealESRGAN_x4plus.pth" | sha256sum -c -

COPY worker/ ${WORKER_HOME}/
RUN chmod +x ${WORKER_HOME}/entrypoint.sh

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /workspace/input /workspace/output /workspace/tmp \
    && chown -R appuser:appuser /workspace ${WORKER_HOME} ${MODELS_DIR}

USER appuser

VOLUME ["/workspace/input", "/workspace/output"]

ENTRYPOINT ["/app/worker/entrypoint.sh"]
