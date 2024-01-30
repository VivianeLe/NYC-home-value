# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
FROM python:3.9.16-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install xgboost
RUN pip install prefect
RUN pip install --upgrade mlflow xgboost
RUN pip install --upgrade pip

COPY ./web_service /app/web_service
WORKDIR /app/web_service

# Make port 8001 available to the world outside this container
EXPOSE 8001

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

