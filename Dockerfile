FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and base tools
RUN apt update && apt install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git curl build-essential ffmpeg libgl1-mesa-glx libglib2.0-0 && \
    ln -sf python3.10 /usr/bin/python && \
    ln -sf python3.10 /usr/bin/python3

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
COPY requirements.txt /app/requirements.txt
WORKDIR /app
# Install base deps
RUN pip install --no-cache-dir -r requirements.txt

# Force correct version of huggingface_hub
RUN pip install --no-cache-dir huggingface_hub==0.22.2 --upgrade

# Copy source code
COPY . /app

CMD [ "python", "train_lora.py" ]
