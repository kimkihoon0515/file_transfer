FROM python:3.9
RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN pip install -r requirements.txt
CMD ["python","data.py","--download_root=/home/jovyan"]