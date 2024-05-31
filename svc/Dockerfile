# docker build -t fschlatt/authorship-svm:0.0.1 .
FROM continuumio/miniconda3

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git curl wget gcc zip unzip
RUN pip3 install pandas tira scikit-learn
ADD run.py /code/run.py
ADD model.joblib /code/model.joblib

ENTRYPOINT [ "python3", "/code/run.py" ]
