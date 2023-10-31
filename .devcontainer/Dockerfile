ARG CUDA_VERSION="1180"
ARG PYTHON="311"

FROM ghcr.io/mjun0812/cuda${CUDA_VERSION}-python${PYTHON}-server:latest

# 環境変数 Versionなど
LABEL maintainer="mjun"

RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

RUN pip install onnx onnxruntime-gpu onnxsim tqdm

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo ssh openssh-client \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

ENTRYPOINT [ "" ]