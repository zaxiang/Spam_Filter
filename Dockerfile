# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
# See https://github.com/ucsd-ets/datahub-docker-stack/wiki/Stable-Tag 
# for a list of the most current containers we maintain
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN apt update
RUN apt-get install build-essential -y
# 3) install packages using notebook user
USER jovyan

# RUN conda install -y scikit-learn

RUN git clone https://github.com/facebookresearch/fastText.git && cd fastText && pip install .

RUN pip install --no-cache-dir --upgrade numpy
RUN pip install --no-cache-dir matplotlib pandas tensorflow pickle4 gensim nltk keras flair keras_preprocessing scipy

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]