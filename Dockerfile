FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY nlpo_toolkit ./nlpo_toolkit
COPY count_corpus_vocabula_local.py ./
COPY config ./config
COPY data ./data

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .

ENTRYPOINT ["nlpo"]
CMD ["count-vocabula", "--project-root", "/app"]
