FROM python:3.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-xcb1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ultralytics torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY app.py app_local.py ./

# Environment variable for Roboflow API key
ENV ROBOFLOW_API_KEY=""

# Create model directory
RUN mkdir -p model

# For GUI applications, we need to set the display
ENV DISPLAY=:0

# Run the application
ENTRYPOINT ["python", "app_local.py"]

# Default parameters that can be overridden at runtime
CMD ["--camera", "0", "--confidence", "0.5", "--download"] 