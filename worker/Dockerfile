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
    # Install pinned runtime deps except basicsr first (reuses torch from base image).
    && grep -v '^basicsr==' requirements.txt > requirements.nobasicsr.txt \
    && pip install --prefer-binary -r requirements.nobasicsr.txt \
    # basicsr is installed without build isolation to avoid pulling a second torch/CUDA stack.
    && pip install --no-build-isolation basicsr==1.4.2 \
    && rm -f requirements.nobasicsr.txt

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
