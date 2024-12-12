# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including libsndfile
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*


# Copy the entire project directory
COPY . /app

# Install specific versions of dependencies
RUN pip install numpy
RUN pip install Cython==0.29.37
RUN pip install spacy==3.0.7

# Install TTS without resolving dependencies
RUN pip install --no-deps tts

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application (updated for xtts.py)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
