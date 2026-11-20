FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

ARG REALESRGAN_X4PLUS_URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
ARG REALESRGAN_X4PLUS_SHA256=4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1
ARG EXPECTED_TORCH_VERSION=2.14.1
ARG EXPECTED_TORCHVISION_VERSION=0.29.1
ARG EXPECTED_CUDA_VERSION=13.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKER_HOME=/app/worker \
    MODELS_DIR=/opt/models \
    WORKER_VENV=/opt/worker-venv

RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
      sed -i -E 's#http://(archive|security)\.ubuntu\.com/ubuntu#https://\1.ubuntu.com/ubuntu#g' /etc/apt/sources.list.d/ubuntu.sources; \
      sed -i -E 's#http://ports\.ubuntu\.com/ubuntu-ports#https://ports.ubuntu.com/ubuntu-ports#g' /etc/apt/sources.list.d/ubuntu.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
      sed -i -E 's#http://(archive|security)\.ubuntu\.com/ubuntu#https://\1.ubuntu.com/ubuntu#g' /etc/apt/sources.list; \
      sed -i -E 's#http://ports\.ubuntu\.com/ubuntu-ports#https://ports.ubuntu.com/ubuntu-ports#g' /etc/apt/sources.list; \
    fi; \
    printf 'Acquire::Retries "5";\nAcquire::http::Timeout "60";\nAcquire::https::Timeout "60";\n' > /etc/apt/apt.conf.d/80-network-resilience; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates curl python3.12-venv; \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${WORKER_HOME}

COPY worker/requirements.txt ./requirements.txt
RUN set -eux; \
    BASE_PY=""; \
    for candidate in python3 python; do \
      if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import torch" >/dev/null 2>&1; then \
        BASE_PY="$candidate"; \
        break; \
      fi; \
    done; \
    if [ -z "$BASE_PY" ]; then \
      echo "No base Python interpreter with preinstalled torch found in this image; cannot keep Blackwell strategy without reinstalling torch." >&2; \
      exit 1; \
    fi; \
    "$BASE_PY" -m venv --system-site-packages ${WORKER_VENV}

ENV PATH=${WORKER_VENV}/bin:${PATH}

RUN python -m pip install --upgrade pip \
    # Install pinned runtime deps except torch/torchvision/realesrgan/basicsr first.
    # Torch/CUDA comes from the base image; avoiding resolver touches prevents heavyweight redownloads.
    && grep -v -E '^(realesrgan|basicsr|torch|torchvision)==' requirements.txt > requirements.core.txt \
    && python -m pip install --prefer-binary -r requirements.core.txt \
    # Ensure runtime torch version and compiled CUDA version expected by this image.
    && python -c "import importlib.util, torch; assert importlib.util.find_spec('packaging') is not None, 'packaging is required for version checks'; from packaging.version import Version; expected=Version('${EXPECTED_TORCH_VERSION}').release; assert Version(torch.__version__).release == expected, f'unexpected torch public version: {torch.__version__}'; expected_cuda=Version('${EXPECTED_CUDA_VERSION}').release; detected_cuda=Version(str(torch.version.cuda)).release; assert tuple(expected_cuda) == tuple(detected_cuda), f'unexpected compiled cuda version: {torch.version.cuda}'" \
    && python -m pip install --no-deps torchvision==${EXPECTED_TORCHVISION_VERSION} \
    # Install BasicSR runtime deps explicitly (excluding torch/torchvision managed above).
    && python -m pip install --prefer-binary addict future lmdb Pillow requests scikit-image scipy tb-nightly tqdm yapf \
    # Install basicsr without build isolation and without dependency resolution.
    && python -m pip install --no-build-isolation --no-deps basicsr==1.4.2 \
    # Patch BasicSR import for torchvision>=0.19 where functional_tensor module path is removed.
    && python -c "import importlib.util; from pathlib import Path; spec = importlib.util.find_spec('basicsr'); assert spec is not None and spec.submodule_search_locations, 'basicsr package path not found'; base = Path(next(iter(spec.submodule_search_locations))); p = base / 'data' / 'degradations.py'; old = 'from torchvision.transforms.functional_tensor import rgb_to_grayscale'; new = 'from torchvision.transforms.functional import rgb_to_grayscale'; s = p.read_text(encoding='utf-8'); assert old in s, f'expected BasicSR patch target not found in {p}'; p.write_text(s.replace(old, new), encoding='utf-8')" \
    # Fail fast if runtime imports still cannot load the patched BasicSR path.
    && python -c "import basicsr; import basicsr.data.degradations; from basicsr.data.degradations import rgb_to_grayscale; print('basicsr_patch_ok')" \
    # Install realesrgan without dependency resolution to avoid re-triggering basicsr install.
    && python -m pip install --no-deps realesrgan==0.3.0 \
    # Verify effective import chain used by runtime.
    && python -c "from basicsr.data.degradations import rgb_to_grayscale; from basicsr.archs.rrdbnet_arch import RRDBNet; from realesrgan import RealESRGANer; print('runtime_import_chain_ok')" \
    && rm -rf /root/.cache/pip /tmp/pip-* \
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
