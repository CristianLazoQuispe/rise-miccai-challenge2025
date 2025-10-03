# ==== Base: PyTorch + CUDA (GPU) ====
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Evita interacciones al instalar paquetes
ARG DEBIAN_FRONTEND=noninteractive

# Zona horaria configurable
ARG CONTAINER_TIMEZONE=Etc/UTC
ENV TZ=${CONTAINER_TIMEZONE}

# Librerías del sistema y utilidades (OpenCV headless, JPEG/PNG, GL)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata ffmpeg libsm6 libxext6 libxrender-dev \
        libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
        libjpeg-dev zlib1g-dev libpng-dev \
        git curl wget unzip ca-certificates && \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python deps
COPY ./requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    # fija versiones críticas que ya probaste
    pip install --no-deps monai==1.5.0 numpy==1.26.4 && \
    pip install -r /tmp/requirements.txt

RUN pip install segmentation-models-pytorch-3d==1.0.2
# todo vive aquí
WORKDIR /my_solution

# Código
COPY ./src ./src
COPY ./csv_creation.py ./csv_creation.py
COPY ./inference_cascade.py ./inference_cascade.py

# Pesos (dentro de la imagen); si prefieres montarlos, mira el compose de abajo
RUN mkdir -p /my_solution/models/fold_models
COPY ./results/fold_models /my_solution/models/fold_models

# Entrypoint dentro de /my_solution
COPY ./entrypoint.sh /my_solution/entrypoint.sh
RUN chmod +x /my_solution/entrypoint.sh

ENV INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    MODEL_DIR=/my_solution/models/fold_models

ENTRYPOINT ["/my_solution/entrypoint.sh"]