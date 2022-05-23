FROM nvidia/cuda:11.0-runtime-ubuntu16.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            git \
            ssh \
            build-essential \
            locales \
            ca-certificates \
            curl \
            unzip \
            vim \
            wget \
            tmux \
            screen \
            pciutils

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Default to utf-8 encodings in python
# Can verify in container with:
# python -c 'import locale; print(locale.getpreferredencoding(False))'
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


RUN pip install sentencepiece==0.1.94
RUN pip install transformers==4.0.0
RUN conda install pytorch==1.7.1 cudatoolkit=11.0 
RUN pip install datasets
RUN pip install wordninja sklearn ipdb nltk english_words tqdm

RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge notebook
RUN pip install gsutil
RUN pip install spicy
RUN pip install boto3

CMD bash
