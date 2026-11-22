FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Train the models at build time so the container starts instantly.
RUN python scripts/train.py

RUN useradd --create-home medbot && chown -R medbot /app
USER medbot

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "gunicorn 'app:create_app()' --bind 0.0.0.0:${PORT} --workers 2 --timeout 60"]
