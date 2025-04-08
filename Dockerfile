# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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

# Install ultralytics
RUN pip install --no-cache-dir ultralytics

# Copy application code
COPY app.py app_local.py ./

# Environment variable for Roboflow API key - hardcoded
ENV ROBOFLOW_API_KEY="XmQnHuLA3FZgcP19fRD8"

# Create model directory
RUN mkdir -p model

# For GUI applications, we need to set the display
ENV DISPLAY=:0

# Run the application
ENTRYPOINT ["python", "app_local.py"]

# Default parameters that can be overridden at runtime
CMD ["--camera", "0", "--confidence", "0.5", "--download"] 