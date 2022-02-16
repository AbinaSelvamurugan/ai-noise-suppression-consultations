FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONSULT_DEVICE=cpu
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY backend/ backend/
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 consult
RUN mkdir -p /app/artifacts/checkpoints && chown -R consult:consult /app/artifacts
USER consult
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
