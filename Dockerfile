FROM tensorflow/tensorflow:2.3.0

COPY . /app

WORKDIR /app

RUN pip3 install -r --user requirements.txt

CMD python3 game/app.py