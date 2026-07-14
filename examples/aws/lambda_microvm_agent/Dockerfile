FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE NOTICE /app/
COPY src /app/src
COPY examples /app/examples
RUN pip install --no-cache-dir ".[aws,egress,postgres,server]"

EXPOSE 800
CMD ["uvicorn", "examples.aws.lambda_microvm_agent.service:app", "--host", "0.0.0.0", "--port", "800"]
