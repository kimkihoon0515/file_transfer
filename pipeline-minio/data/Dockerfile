FROM python:3.9
RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN pip install -r requirements.txt
CMD ["python","data_download.py","--download_root=/MNIST_DATA","--minio_bucket=mlpipeline"]