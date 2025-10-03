FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use Gunicorn as entrypoint (production WSGI server)
CMD ["gunicorn", "--workers=2", "--threads=4", "--timeout=600", "--bind=0.0.0.0:8001", "server:app"]
