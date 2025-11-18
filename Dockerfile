# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /norm-fullstack

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Environment variables (should be set at runtime)
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV API_KEY=$API_KEY

# Copy the content of the local src directory to the working directory
COPY ./app /norm-fullstack/app
COPY ./docs /norm-fullstack/docs

# Command to run on container start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]