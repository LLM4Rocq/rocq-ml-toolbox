FROM theostos/rocq-lsp:9.0

USER rocq
WORKDIR /home/rocq

RUN curl -s -L -o miniconda_installer.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash miniconda_installer.sh -b -c -p /home/rocq/miniconda \
 && rm miniconda_installer.sh

ENV PATH=/home/rocq/miniconda/bin:$PATH

RUN /home/rocq/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && /home/rocq/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN /home/rocq/miniconda/bin/conda create -y -n rocq-ml python=3.12

ENV PATH=/home/rocq/miniconda/envs/rocq-ml/bin:$PATH

RUN git clone https://github.com/LLM4Rocq/pytanque.git pytanque-repo \
 && cd pytanque-repo \
 && /home/rocq/miniconda/envs/rocq-ml/bin/pip install -e .

RUN git clone https://github.com/LLM4Rocq/rocq-ml-toolbox.git \
 && cd rocq-ml-toolbox \
 && git switch develop \
 && /home/rocq/miniconda/envs/rocq-ml/bin/pip install -e .[server]
