# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirement file first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY app/ .

# Expose Flask port
EXPOSE 40404

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=40404

# Run Flask app
CMD ["flask", "run"]
