# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data logs checkpoints

# Copy the application code
COPY models/ models/
COPY utils/ utils/
COPY src/ src/

# Set up entry point script
COPY docker-entrypoint.sh .
# Set permissions for entrypoint script (moved before user creation)
RUN chmod +x docker-entrypoint.sh

# Create a non-root user
RUN useradd -m -u 1000 model-user
RUN chown -R model-user:model-user /app
USER model-user

# Copy the model checkpoint if it exists (after user creation)
COPY --chown=model-user:model-user checkpoints/ checkpoints/

# Expose port for API
EXPOSE 8000

# Set the entry point
ENTRYPOINT ["./docker-entrypoint.sh"] 