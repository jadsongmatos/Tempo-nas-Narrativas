# Use uma imagem base oficial do Python
FROM python:3.9.20-slim
#FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
#FROM pytorch/manylinux-cpu:latest
#FROM python:3.9.20-alpine3.20

# Defina variáveis de ambiente para otimizar o build
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instale dependências do sistema
#RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Crie diretório para a aplicação
WORKDIR /app

# Copie o restante do código
COPY requirements.txt /app/

# Crie diretório para o cache do pip
RUN mkdir -p /mnt/meu_btrfs/pip_cache

# Defina a variável de ambiente para o cache do pip
ENV PIP_CACHE_DIR=/mnt/meu_btrfs/pip_cache

# Copie e instale as dependências Python
COPY requirements.txt /app/
#RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código
COPY process.py /app/

# Defina o comando padrão
CMD ["python", "process.py"]

