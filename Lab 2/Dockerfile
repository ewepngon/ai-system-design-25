FROM python:3.9-slim

# We need to upgrade Python 3 to at least 3.7 because pkgconf has format strings! yuck
# RUN apt-get update -y && \
#    apt-get install -y python3-pip python3-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app2/requirements.txt

WORKDIR /app2

RUN pip3 install -r requirements.txt

COPY . /app2

ENV FLASK_APP=hello_flask
ENV FLASK_ENV=development

CMD [ "python3", "hello_flask.py", "4000" ]

