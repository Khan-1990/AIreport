# Start with a standard NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Install Python 3.11, pip, git, and aria2 for torrent downloads
RUN apt-get update && \
    apt-get install -y python3.11 python3-pip git git-lfs aria2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Set the application directory
WORKDIR /app

# Copy just the requirements file to leverage Docker's build cache
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set the default command to open a bash shell
CMD ["/bin/bash"]
