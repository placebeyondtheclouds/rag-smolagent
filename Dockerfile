FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1

ENV GRADIO_ANALYTICS_ENABLED="False"
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0
ENV DISABLE_TELEMETRY=1
ENV DO_NOT_TRACK=1
ENV HF_HUB_DISABLE_IMPLICIT_TOKEN=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# permissions and nonroot user for tightened security
RUN adduser --disabled-password nonroot
RUN mkdir /app/ && chown -R nonroot:nonroot /app
WORKDIR /app
USER nonroot

COPY --chown=nonroot:nonroot requirements.txt .

# venv
ENV VIRTUAL_ENV=/app/venv

# python setup
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN printf "[global]\n\
timeout = 10\n\
index-url = https://pypi.tuna.tsinghua.edu.cn/simple\n\
\n\
trusted-host =\n\
  pypi.tuna.tsinghua.edu.cn\n\
  pypi.org\n\
  pypi.python.org\n\
  mirrors.aliyun.com\n\
  pypi.mirrors.ustc.edu.cn\n" >/app/venv/pip.conf

# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache pip install --no-cache-dir --timeout=100 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --find-links https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html torch===2.2.2+cu118 torchaudio===2.2.2+cu118
RUN --mount=type=cache,target=/root/.cache pip install --no-cache-dir --timeout=100 -r requirements.txt
# RUN pip install -r requirements.txt

COPY --chown=nonroot:nonroot ./ingest_pdfs.py .
COPY --chown=nonroot:nonroot ./smolagent_rag.py .
COPY --chown=nonroot:nonroot ./.env .

EXPOSE 7860
ENTRYPOINT ["python"]
CMD ["smolagent_rag.py"]
# CMD ["/usr/bin/tail", "-f", "/dev/null"]
