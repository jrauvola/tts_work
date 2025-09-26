FROM python:3.11-slim

WORKDIR /app
COPY gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY protos/gen/python /app/protos/gen/python
COPY gateway /app/gateway
COPY tts_common /app/tts_common
COPY frontend /app/frontend

ENV PYTHONPATH=/app/protos/gen/python:/app
EXPOSE 8000
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]

