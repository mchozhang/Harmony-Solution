
# Configure virtual environment
FROM python:3.8-buster
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

# Run the app
CMD python app.py