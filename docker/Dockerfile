FROM python:3.10.6
WORKDIR /ML_FAST_API
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . /ML_FAST_API
ENTRYPOINT [ "streamlit", "run" ]
CMD ["app.py"]
