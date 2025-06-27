# Use a stable official NVIDIA CUDA base image with PyTorch
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set environment variables to ensure non-interactive setup
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Update package lists, install git, and then install Python packages
# This also cleans up apt cache to reduce image size.
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# This command will be run by default when the container starts.
# We will override this when running specific jobs (like training or inference).
CMD ["/bin/bash"]
