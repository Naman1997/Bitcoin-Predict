FROM tensorflow/tensorflow
RUN apt-get install -y python3.7
RUN apt-get clean
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    python3-pip
RUN mkdir -p app
RUN python3 -m pip install -U pip
COPY requirements.txt /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN pip3 install tensorflow && \
    pip3 install numpy scipy scikit-learn pillow pandas sklearn matplotlib seaborn jupyter pyyaml h5py && \
    pip3 install keras --no-deps && \
    pip3 install opencv-python && \
    pip3 install imutils
COPY . .
RUN pip3 install -r req-2.txt
CMD python3 app.py
