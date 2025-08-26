FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install HF dependencies
RUN pip install --no-cache-dir \
    torch==2.1.1+cpu \
    transformers==4.36.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy app
COPY . .

# Create cache directory
RUN mkdir -p /app/hf_cache
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV ENABLE_MONITORING=true

EXPOSE 7860

CMD ["python", "app.py"]