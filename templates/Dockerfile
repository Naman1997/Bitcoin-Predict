FROM ubuntu:18.04

RUN apt-get update && apt-get install -q -y --fix-missing \
	wget \
	curl \
	net-tools \
	python-pip

RUN apt-get clean

COPY . .

RUN pip install -r requirements.txt

RUN python3 app.py