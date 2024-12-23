ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}
RUN mkdir -p /experiment
RUN python -m pip install poetry
VOLUME "/data"
ENV DATA_DIR=/data
WORKDIR /experiment
COPY poetry.lock .
COPY pyproject.toml .
RUN poetry install
COPY . /experiment
ENV OWNER=1000:1000
CMD export OUTPUT_DIR=$DATA_DIR/$(date +%Y-%m-%d-%H-%M-%S)-$(hostname) && \
    mkdir -p $OUTPUT_DIR && \
    poetry run python src/main.py | tee $OUTPUT_DIR/output.log && \
    chown -R $OWNER $DATA_DIR