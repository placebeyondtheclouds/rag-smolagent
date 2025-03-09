FROM python:3.10-slim

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

# set up mirrors for debian bookworm distribution
RUN <<EOF
tee /etc/apt/sources.list <<-'EOF2'
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware
deb https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware
EOF2
EOF

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

COPY --chown=nonroot:nonroot ./pip.conf /app/venv/pip.conf

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
# CMD ["tail", "-f", "/dev/null"]
