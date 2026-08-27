FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# Whisper decodes through libav; ffmpeg covers the container formats browsers
# actually produce (m4a, webm, ogg) as well as wav and mp3.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-asr.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-asr.txt

COPY app ./app
COPY scripts ./scripts
COPY tests/fixtures ./tests/fixtures

# Recordings, the SQLite database and the downloaded Whisper model all live
# under /data. Mount a persistent volume over it - without one, every restart
# loses every record.
ENV DATA_DIR=/data \
    DATABASE_URL=sqlite:////data/manager_convo.sqlite3 \
    HF_HOME=/data/models
VOLUME ["/data"]

EXPOSE 8000

# Most managed hosts inject $PORT; fall back to 8000 for a plain `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
