ARG CUDA_VERSION="1180"
ARG PYTHON="311"

FROM ghcr.io/mjun0812/cuda${CUDA_VERSION}-python${PYTHON}-server:latest

# 環境変数 Versionなど
LABEL maintainer="mjun"

RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

RUN pip install onnx onnxruntime-gpu onnxsim tqdm
