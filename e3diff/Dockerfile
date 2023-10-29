FROM tensorflow/tensorflow:gpu
COPY pyproject.toml .
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install
