FROM python:3.12-slim

WORKDIR /app

# Зависимости устанавливаем отдельным слоем — кешируется если requirements.txt не менялся
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходники
COPY app/ ./app/
COPY indexer/ ./indexer/

EXPOSE 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
