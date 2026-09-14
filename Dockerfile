FROM python:3.12-slim

# libgomp is required by LightGBM at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY . .

# Build the cohort and train at image-build time so the container starts ready.
# The seed is fixed, so the model baked into the image is the one the evaluation
# in docs/evaluation.md describes.
RUN python scripts/generate_data.py --patients 600 \
 && python ml/train.py \
 && python ml/evaluate.py --skip-ablation

EXPOSE 8000
ENV DOSESENSE_PATIENT_LIMIT=0

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if b'\"status\":\"ok\"' in urllib.request.urlopen('http://127.0.0.1:8000/api/health').read() else 1)"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

