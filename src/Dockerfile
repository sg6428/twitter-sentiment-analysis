# python runtime as a parent image
FROM python:3.10-slim

# setting the working directory in the container
WORKDIR /app

# copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of the application files
COPY . .

# expose the FastAPI default port
EXPOSE 8000
