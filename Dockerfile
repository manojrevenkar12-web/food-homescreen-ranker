# SageMaker-compatible inference image: GET /ping and POST /invocations on port 8080, model in /opt/ml/model.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd --uid 10001 --create-home app
WORKDIR /app
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt
COPY config ./config
COPY src ./src
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 FHR_BUNDLE=/opt/ml/model
USER 10001
EXPOSE 8080
ENTRYPOINT ["python", "-c", "from fhr.serve.app import main; main()"]
