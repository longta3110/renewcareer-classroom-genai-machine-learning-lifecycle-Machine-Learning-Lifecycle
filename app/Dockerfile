FROM python:3.11-slim
WORKDIR /app
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src /app/src
COPY app /app/app
COPY models/transformer /app/models/transformer
ENV PORT=5000
EXPOSE 5000
HEALTHCHECK --interval=30s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"
CMD ["python", "-m", "app.api"]
