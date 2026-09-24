 
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY data/raw ./data/raw

ENV FLASK_APP=app.api
EXPOSE 5000

CMD ["python", "app/api.py"]
