FROM tensorflow/tensorflow:2.18.0-gpu
RUN pip3 install tensorflow-datasets
RUN pip3 install silence-tensorflow
RUN apt update
RUN apt install -y flatbuffers-compiler git
RUN git clone https://github.com/USNA-CE-ACN/DeepLearn.git