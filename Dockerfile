# Start with a stable official NVIDIA CUDA base image that includes PyTorch
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Update package lists, install git and git-lfs for model downloading
RUN apt-get update && apt-get install -y git git-lfs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the Python packages using pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    git lfs install

# Copy the rest of your application code (train.py, inference.py, etc.)
COPY . .

# Set the default command to open a bash shell, which will be overridden by jobs
CMD ["/bin/bash"]
