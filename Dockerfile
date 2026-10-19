FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_PATH=/app/models/best.pt HOST=0.0.0.0 PORT=5000
WORKDIR /app

# CPU-only PyTorch keeps the image small (~1 GB instead of ~5 GB)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY foodai ./foodai
COPY webapp ./webapp
COPY data/nutrition.json ./data/nutrition.json

# The trained model is mounted, not baked in:
#   docker run -p 5000:5000 -v "$(pwd)/models:/app/models" food-classifier
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/health').status==200 else 1)"

# one worker: the model is loaded once per worker; threads share it safely (inference is lock-protected)
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "webapp.app:app"]
