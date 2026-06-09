# XAU Trading Bot — alerts-only. Runs the live alert loop and/or the web chart.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Jerusalem

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Default: run the live alert loop. docker-compose overrides command per service.
CMD ["python", "scripts/live_alerts.py"]
