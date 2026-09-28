FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0

RUN pip3 install \
        "numpy<2" basicsr realesrgan opencv-python-headless boto3 runpod

# basicsr 1.4.2 imports a module removed in torchvision>=0.17
RUN sed -i \
    's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py || true

RUN mkdir -p /models \
    && python3 -c "import urllib.request as u; u.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth','/models/realesr-general-x4v3.pth')" \
    && python3 -c "import urllib.request as u; u.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth','/models/RealESRGAN_x4plus.pth')"

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
