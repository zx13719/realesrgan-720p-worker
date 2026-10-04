# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Ubuntu 24.04 ships Python 3.12 + ffmpeg 6.1 (much newer than 22.04's ffmpeg 4.4)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 torchvision==0.22.0

# spandrel replaces basicsr+realesrgan: no torchvision monkey-patch, fewer deps
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# Bake all weights into the image and verify spandrel can read each architecture
RUN mkdir -p /models \
    && python3 -c "import urllib.request as u; u.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth','/models/realesr-general-x4v3.pth')" \
    && python3 -c "import urllib.request as u; u.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth','/models/RealESRGAN_x2plus.pth')" \
    && python3 -c "import urllib.request as u; u.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth','/models/RealESRGAN_x4plus.pth')" \
    && python3 - <<'PY'
from spandrel import ModelLoader
for name, path in {
    "realesr-general-x4v3": "/models/realesr-general-x4v3.pth",
    "RealESRGAN_x2plus": "/models/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus": "/models/RealESRGAN_x4plus.pth",
}.items():
    d = ModelLoader().load_from_file(path)
    print(name, "->", d.architecture.name, "x%d" % d.scale)
PY

ARG VCS_REF=unknown
ENV IMAGE_REVISION=$VCS_REF
LABEL org.opencontainers.image.revision=$VCS_REF

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
