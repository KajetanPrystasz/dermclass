FROM python:3.6.9-slim
FROM continuumio/miniconda3

SHELL [ "/bin/bash", "--login", "-c" ]

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' dermclass_api_user


WORKDIR /opt/dermclass_api

ARG PIP_EXTRA_INDEX_URL
ENV FLASK_APP run.py

# Install requirements, including from Gemfury
ADD ./src/dermclass_api /opt/dermclass_api/

RUN pip install --upgrade pip

RUN conda env create -f /opt/dermclass_api/environment.yml
SHELL ["conda", "run", "-n", "dermclass_api", "/bin/bash", "--login", "-c"]

RUN chown -R dermclass_api_user:dermclass_api_user ./
RUN chmod +x /opt/dermclass_api/run.sh

#TODO: Fix user for conda
#USER dermclass_api_user

EXPOSE 5000

ENTRYPOINT ["conda", "run", "-n", "dermclass_api", "bash", "./run.sh"]