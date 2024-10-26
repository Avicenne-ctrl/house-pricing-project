FROM python:3.12-slim    
WORKDIR /app

RUN pip install --upgrade pip

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["python", "app.py"]

