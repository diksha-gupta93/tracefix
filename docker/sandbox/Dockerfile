FROM python:3.14-slim

RUN python -m pip install --no-cache-dir "pytest>=8.4,<9"

RUN groupadd --gid 10001 tracefix \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent tracefix

USER 10001:10001
WORKDIR /tracefix/workspace

ENTRYPOINT ["python", "-c", "import os,shutil,sys; shutil.copytree('/tracefix/input','/tracefix/workspace/repository'); os.chdir('/tracefix/workspace/repository'); os.execv(sys.executable,(sys.executable,'-m','pytest',*sys.argv[1:]))"]
