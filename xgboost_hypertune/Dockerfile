
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-11

# Avoiding crash due to scikit-learn and xgboost installation 
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt
CMD ["python", "app.py"]
