FROM python:3.14-slim

RUN python -m pip install --no-cache-dir \
    "ruff>=0.12,<0.13" \
    "mypy>=1.17,<2" \
    "bandit>=1.9,<2"

RUN groupadd --gid 10001 tracefix \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent tracefix

COPY docker/static-checks/runner.py /opt/tracefix/static_runner.py

USER 10001:10001
WORKDIR /tracefix/workspace

ENTRYPOINT ["python", "/opt/tracefix/static_runner.py"]
