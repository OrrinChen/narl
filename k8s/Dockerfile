FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

WORKDIR /narl
COPY requirements-cpu.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.txt \
    && python -c 'import torch; assert torch.version.cuda is None'

COPY algos.py core.py model.py serve.py tasks.py train.py utils.py ./
COPY experiments/cpu_baseline.py experiments/kv_replay.py ./experiments/
COPY configs/narl_cpu_baseline.yaml ./configs/

ARG NARL_SOURCE_COMMIT
ENV NARL_SOURCE_COMMIT=$NARL_SOURCE_COMMIT \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false
