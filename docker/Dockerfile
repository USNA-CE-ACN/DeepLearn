FROM tensorflow/tensorflow:2.18.0-gpu
RUN pip3 install tensorflow-datasets
RUN pip3 install silence-tensorflow
RUN apt update
RUN apt install -y flatbuffers-compiler git emacs
RUN git clone https://github.com/USNA-CE-ACN/DeepLearn.git
COPY downloader.py /usr/local/lib/python3.11/dist-packages/tensorflow_datasets/core/download/downloader.py