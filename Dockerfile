FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Architecture detection for platform-specific installations
RUN dpkg --print-architecture > /tmp/arch && \
    if grep -q "arm\|aarch" /tmp/arch; then \
    echo "ARM architecture detected" && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && apt-get install -y gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "Installing EdgeTPU support" ; \
    fi

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk-3-0 \
    python3-dev \
    && if grep -q "arm\|aarch" /tmp/arch; then \
       apt-get install -y \
       libedgetpu1-std \
       python3-pycoral \
       libatlas-base-dev \
       libjpeg-dev \
       libopenjp2-7 \
       ; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install platform-specific packages
RUN if grep -q "arm\|aarch" /tmp/arch; then \
    echo "Installing Raspberry Pi specific packages" && \
    pip install --no-cache-dir \
    https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl || true; \
    else \
    echo "Installing PC specific packages" && \
    pip install --no-cache-dir \
    pyrealsense2; \
    fi

# Install UI-related packages
RUN pip install --no-cache-dir \
    flask \
    flask-socketio \
    simple-websocket

# Copy the application code
COPY . .

# Create directories if they don't exist
RUN mkdir -p models
RUN mkdir -p config
RUN mkdir -p ui/static/images

# Make sure ui directory exists with proper permissions
RUN mkdir -p ui/static/css ui/static/js ui/templates

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV FLASK_APP=ui/server.py

# Expose port for UI
EXPOSE 9029

# Default command - will be overridden by docker-compose
CMD ["python", "main.py", "--config", "config/config.json"] 