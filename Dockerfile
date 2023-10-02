FROM python:latest

WORKDIR /usr/workspace/resin

COPY . .

RUN python3 -m pip install --upgrade poetry
RUN poetry install
RUN poetry build

EXPOSE 8000

CMD [ "poetry", "run", "resin", "start" ]
