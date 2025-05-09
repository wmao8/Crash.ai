FROM python:3.9-slim

WORKDIR /app_US_accidents

COPY . /app_US_accidents

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app_US_accidents.py", "--server.port=8501", "--server.enableCORS=false"]
