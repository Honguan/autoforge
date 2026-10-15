# AutoForge — CPU image (classic ML packs, GUI, MCP, REST). For NVIDIA GPUs use Dockerfile.cuda.
#   docker run --rm -p 127.0.0.1:8800:8800 -v autoforge:/data ghcr.io/honguan/autoforge:cpu
ARG BASE=python:3.12-slim
FROM ${BASE}
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 AUTOFORGE_HOME=/data
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl
VOLUME /data
EXPOSE 8800 8000
ENTRYPOINT ["autoforge"]
CMD ["gui", "--host", "0.0.0.0", "--no-open"]
