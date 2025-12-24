# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
# We install torch CPU version to keep the image smaller
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the API
CMD ["python", "main.py"]