# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
RUN mkdir project
WORKDIR /project

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire content of the local directory into the container
COPY . .

# Specify the command to run your application
# Adjust this to the script you want to run by default
# CMD ["python", "make_prediction.py"]
