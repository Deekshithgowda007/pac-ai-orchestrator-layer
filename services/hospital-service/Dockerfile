FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    dcmtk \
    libopenjp2-7 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install fastapi uvicorn psycopg2-binary requests python-multipart pynetdicom pydicom
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]