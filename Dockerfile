# Imagen base com Python 3.12 no Ubuntu 22.04
FROM python:3.12-slim

# Evita prompts interativos durante o apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SDL_VIDEODRIVER=dummy

# Instalar dependencias do sistema necessarias para Pygame, Ray e PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    && rm -rf /var/lib/apt/lists/*

# Definir diretorio de trabalho
WORKDIR /app

# Copiar requirements e instalar dependencias Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o código do projeto para o container
COPY . /app

# Instalar o pacote overcooked_marl_env em modo editável
RUN pip install -e .

# Comando padrao para executar o experimento
CMD ["bash", "experiments/training/run_experiment_cycle.sh"]
