FROM python:3.12-slim

# Evita prompts interativos e define charset
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Instala dependências de sistema para matplotlib, pandas etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório da aplicação
WORKDIR /app

# Copia dependências e instala
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o código
COPY app/ ./app/

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Comando principal
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=localhost"]
