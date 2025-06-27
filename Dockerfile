# Use an official NVIDIA CUDA base image which includes GPU drivers and PyTorch
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python libraries, huggingface_hub for downloading, and git-lfs
RUN apt-get update && apt-get install -y git git-lfs && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    git lfs install

# Copy the rest of the application code into the container
COPY . .

# This command will be run by default when the container starts.
# We will override this when running specific jobs (like training or inference).
CMD ["/bin/bash"]
