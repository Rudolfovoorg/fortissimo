FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://www.piwheels.org/simple -r requirements.txt


COPY app.py prep_data.py ./
ENV TFLITE_THREADS=2

EXPOSE 8050
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8050"]
