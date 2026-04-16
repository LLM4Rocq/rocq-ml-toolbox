ARG BASE_IMAGE=theostos/coq-lsp:8.20
FROM ${BASE_IMAGE}

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl gpg lsb-release && \
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/redis.list && \
    printf 'Package: redis redis-server redis-sentinel redis-tools\nPin: version 6:7.4.*\nPin-Priority: 1001\n' \
      > /etc/apt/preferences.d/redis && \
    apt-get update && \
    apt-get install -y --no-install-recommends redis redis-server redis-tools && \
    rm -rf /var/lib/apt/lists/*

USER coq
WORKDIR /home/coq

RUN curl -s -L -o miniconda_installer.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash miniconda_installer.sh -b -c -p /home/coq/miniconda \
 && rm miniconda_installer.sh

ENV PATH=/home/coq/miniconda/bin:$PATH

RUN /home/coq/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && /home/coq/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN /home/coq/miniconda/bin/conda create -y -n rocq-ml python=3.12

ENV PATH=/home/coq/miniconda/envs/rocq-ml/bin:$PATH

RUN git clone https://github.com/LLM4Rocq/pytanque.git pytanque-repo \
 && cd pytanque-repo \
 && /home/coq/miniconda/envs/rocq-ml/bin/pip install -e .

RUN git clone https://github.com/LLM4Rocq/rocq-ml-toolbox.git \
 && cd rocq-ml-toolbox \
 && /home/coq/miniconda/envs/rocq-ml/bin/pip install -e .[server]
