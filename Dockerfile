FROM pklinker/coral-python:1

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    libopencv-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python libraries
RUN pip3 install --no-cache-dir \
    numpy \
    pillow \
    pycoral \
    pytesseract \
    scikit-learn \
    matplotlib \
    pynetworktables \
    cscore \
    opencv-contrib-python \
    tqdm \
    imgaug

# Set up directories for the application
WORKDIR /app
RUN mkdir -p /app/calibration /app/training /app/models /app/data

# Copy project files
COPY vision_processing.py /app/
COPY labels.txt /app/
COPY README.md /app/

# This will be the volume mount point for the USB device
VOLUME /dev/bus/usb

# Make the scripts executable
RUN chmod +x /app/vision_processing.py

# Set environment variables
ENV PYTHONPATH=/app

# Set entry point to run the vision processing script
CMD ["python3", "/app/vision_processing.py"] 