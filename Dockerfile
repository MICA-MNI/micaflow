# Start from a standard python base image, or a neuroimaging standard like an ubuntu base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies if your pipeline utilities need them
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your codebase into the container
COPY . /app/

# Install the python package and its dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Pre-download models to bake them into the image
# This prevents the CLI from needing internet access at runtime
RUN python -c "from micaflow.cli import check_and_download_models; check_and_download_models()"

# Set the default entrypoint to your CLI
ENTRYPOINT ["micaflow"]
CMD ["--help"]