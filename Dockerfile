FROM python:3.7.9-slim
FROM continuumio/miniconda3

RUN apt update -y
RUN apt install -y libgl1-mesa-glx

WORKDIR /opt/dermclass_api

ARG PIP_EXTRA_INDEX_URL
ENV FLASK_APP run.py

ADD ./src/dermclass_api /opt/dermclass_api/

RUN pip install --upgrade pip

RUN conda env create -f /opt/dermclass_api/environment.yml

RUN useradd -m dermclass_api_user
RUN chown -R dermclass_api_user ./
RUN chown -R dermclass_api_user /opt/conda/envs/dermclass_api
RUN chmod +x /opt/dermclass_api/run.sh
USER dermclass_api_user

CMD ["conda", "run", "-n", "dermclass_api", "bash", "./run.sh"]