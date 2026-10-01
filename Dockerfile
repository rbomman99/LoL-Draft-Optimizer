# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your Python scripts into the container
COPY . .

# Run a training worker; FastAPI will be added separately.
CMD ["python", "draft_optimizer.py"]
