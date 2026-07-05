FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STANZA_RESOURCES_DIR=/opt/stanza_resources

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY nlpo_toolkit ./nlpo_toolkit
COPY utils.py ./utils.py

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

ARG DOWNLOAD_STANZA_MODELS=true
RUN if [ "$DOWNLOAD_STANZA_MODELS" = "true" ]; then \
      python -c "import os, stanza; stanza.download('la', package='perseus', model_dir=os.environ['STANZA_RESOURCES_DIR'])"; \
    fi

VOLUME ["/workspace", "/opt/stanza_resources"]
WORKDIR /workspace

ENTRYPOINT ["nlpo"]
CMD ["--help"]
