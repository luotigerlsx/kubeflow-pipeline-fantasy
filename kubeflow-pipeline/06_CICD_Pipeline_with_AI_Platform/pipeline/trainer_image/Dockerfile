FROM tensorflow/tensorflow:1.15.0-py3
RUN pip install -U fire cloudml-hypertune pandas google-api-python-client scikit-learn gcsfs
WORKDIR /app
COPY . /app

ENTRYPOINT ["python", "train.py"]
