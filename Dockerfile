FROM tensorflow/tensorflow:latest

COPY . /harmony-solution

RUN pip3 install -r requirements.txt

CMD python3