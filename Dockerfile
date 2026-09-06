FROM python:3.11-slim

WORKDIR /app

# CPU-only torch/torchvision wheels — the CUDA build would blow up the image size
# and no GPU is available inside these containers anyway.
RUN pip install --no-cache-dir flask==3.0.3 requests==2.32.3 && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 torchvision==0.18.1

COPY client ./client

# Bake the MNIST download into the image layer so all containers start with the
# data already on disk instead of racing each other to download it at runtime.
ENV DATA_DIR=/data
RUN python -c "from torchvision import datasets; \
    datasets.MNIST('/data', train=True, download=True); \
    datasets.MNIST('/data', train=False, download=True)"

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

CMD ["python", "-m", "client.server"]
