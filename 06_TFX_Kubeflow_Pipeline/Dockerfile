FROM tensorflow/tfx:0.21.2
RUN pip install -U tensorflow-serving-api==1.15 tensorflow==1.15
RUN mkdir modules
COPY  transform_train.py modules/