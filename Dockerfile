FROM tensorflow/tensorflow:latest-py3
RUN apt-get clean
RUN apt-get update && apt-get install -y \
    python3-pip
RUN mkdir -p app
COPY requirements.txt /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN pip3 install glob2
CMD python3 app.py
