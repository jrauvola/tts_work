FROM python:3.11-slim

WORKDIR /app
COPY textproc/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY textproc /app/textproc

EXPOSE 8001
CMD ["uvicorn", "textproc.app:app", "--host", "0.0.0.0", "--port", "8001"]

