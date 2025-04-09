FROM debian:buster

# Install system dependencies
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y curl gnupg ca-certificates zlib1g-dev libjpeg-dev git apt-utils \
    usbutils libusb-1.0-0 libusb-1.0-0-dev python3-dev python3-numpy python3-matplotlib \
    python3-opencv libopencv-dev cmake build-essential \
    libatlas-base-dev libblas-dev liblapack-dev

# Add Coral repository and install Edge TPU runtime
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install Python and Coral libraries
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y python3 python3-pip
RUN apt-get install -y libedgetpu1-legacy-std python3-edgetpu
RUN apt-get install -y python3-pycoral python3-tflite-runtime

# Install additional Python packages
RUN pip3 install pillow
# We're using system-provided OpenCV instead of pip version

# Set up working directory
WORKDIR /app

# Copy requirements and install additional Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make main script executable
RUN chmod +x main.py

# Add PYTHONPATH to ensure src is in the module search path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Command to run when container starts
CMD ["python3", "main.py", "--help"] 