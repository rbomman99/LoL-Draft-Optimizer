# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your Python scripts into the container
COPY . .

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to run the FastAPI microservice
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]