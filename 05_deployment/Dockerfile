FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY ["pyproject.toml", "uv.lock", "./"]

RUN uv sync --locked

COPY ["predict.py", "model_c=0.01.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
