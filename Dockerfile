FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose port for FastAPI
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
