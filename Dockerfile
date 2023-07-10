FROM python:3.8-slim

RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y sox libsndfile1 libsamplerate0-dev
RUN apt-get update && apt-get install -y git make nasm pkg-config libx264-dev libxext-dev libxfixes-dev zlib1g-dev

RUN apt-get --assume-yes install \
    git \
    curl \
    xz-utils\
    libpq-dev \
    gcc \
    g++ \
    openssh-client

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./ /code

EXPOSE 5000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"]
