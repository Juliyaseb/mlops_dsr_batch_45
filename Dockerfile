FROM python:3.12-slim
WORKDIR /code

#copy requirements file into the working dir
COPY ./requirements.txt /code/requirements.txt

COPY ./app /code/app

ENV WANDB_ORG = ‘’
ENV WANDB_PROJECT = ‘’
ENV WANDB_MODEL_NAME = ‘’
ENV WANDB_MODEL_VERSION = ‘’
ENV WANDB_API_KEY = ‘’

EXPOSE 8080
CMD [“fastapi”, “run”,  “app/main.py”, “--port”, “8080”, “--host”, “--reload”]