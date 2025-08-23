FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY electricitydemandforecaster/ /app/electricitydemandforecaster

WORKDIR /app/electricitydemandforecaster

RUN pip install --no-cache-dir .

WORKDIR /app

COPY api /app/api

COPY models /app/models

COPY data /app/data

RUN pip install --no-cache-dir fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
