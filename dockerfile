# Use the official Python image from Docker Hub as the base image
FROM python:3.8-slim

# Set working directory
WORKDIR /usr/src/app

# Copy the Python script to the working directory
COPY sparse_recommender.py .

# Upgrade pip and install numpy
RUN pip install --upgrade pip && pip install numpy

# Run the Python script
CMD [ "python", "./sparse_recommender.py" ]
