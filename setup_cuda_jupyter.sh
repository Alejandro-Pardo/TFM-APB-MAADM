#!/bin/bash

# CUDA JupyterLab Setup Script - NVIDIA Container Edition
# Uses NVIDIA's official PyTorch container with CUDA 12.2+ support
# Fixed cache permissions for Hugging Face models

# Color variables
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check prerequisites
check_prerequisites() {
    echo -e "${CYAN}Checking prerequisites...${NC}"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker compose &> /dev/null; then
        echo -e "${RED}Error: Docker Compose is not installed. Please install Docker Compose first.${NC}"
        exit 1
    fi

    # Check for NVIDIA Docker runtime
    if ! docker info | grep -q nvidia; then
        echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected.${NC}"
        echo -e "${YELLOW}Please ensure you have nvidia-container-toolkit installed.${NC}"
        echo -e "${YELLOW}Installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
        read -p "Continue anyway? (y/N): " continue_choice
        if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check CUDA version on host
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}NVIDIA SMI detected:${NC}"
        nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader,nounits
    else
        echo -e "${YELLOW}Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed.${NC}"
    fi
}

# Function to validate port
validate_port() {
    local port=$1
    
    if ! [[ $port =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Port must be a number${NC}"
        return 1
    fi
    
    if [ "$port" -lt 8000 ] || [ "$port" -gt 9999 ]; then
        echo -e "${RED}Port must be between 8000 and 9999${NC}"
        return 1
    fi
    
    # Check if port is already in use
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}Port $port is already in use${NC}"
        return 1
    fi
    
    return 0
}

# Function to generate secure password
generate_password() {
    openssl rand -base64 12 | tr -d "=+/" | cut -c1-12
}

# Function to check password strength
check_password_strength() {
    local password="$1"
    local length=${#password}
    
    if [[ $length -lt 8 ]]; then
        echo -e "${RED}Password too short (minimum 8 characters)${NC}"
        return 1
    fi
    
    return 0
}

# Main configuration function
get_configuration() {
    echo -e "\n${CYAN}CUDA JupyterLab Configuration (NVIDIA Container)${NC}"
    echo -e "${YELLOW}======================================================${NC}"
    
    # Get container name
    DEFAULT_CONTAINER_NAME="pardo-tfm-jupyter"
    read -p "Container name (default: ${DEFAULT_CONTAINER_NAME}): " CONTAINER_NAME
    CONTAINER_NAME=${CONTAINER_NAME:-$DEFAULT_CONTAINER_NAME}
    
    # Get user info
    USERNAME="pardo"
    USER_UID="1012"
    USER_GID="1013"
    echo -e "Will run as user: ${CYAN}$USERNAME${NC} (UID: $USER_UID, GID: $USER_GID)"
    
    # Get port
    while true; do
        read -p "JupyterLab port (8000-9999, default: 8977): " JUPYTER_PORT
        JUPYTER_PORT=${JUPYTER_PORT:-8977}
        if validate_port "$JUPYTER_PORT"; then
            break
        fi
    done
    
    # Get password
    read -p "Generate secure password? (Y/n): " gen_password
    if [[ "$gen_password" =~ ^[Nn]$ ]]; then
        while true; do
            read -s -p "JupyterLab password: " JUPYTER_PASSWORD
            echo
            if check_password_strength "$JUPYTER_PASSWORD"; then
                break
            fi
        done
    else
        JUPYTER_PASSWORD=$(generate_password)
        echo -e "Generated password: ${CYAN}$JUPYTER_PASSWORD${NC}"
        echo -e "${YELLOW}Please save this password!${NC}"
    fi
    
    # Get workspace directory
    DEFAULT_WORKSPACE="./workspace"
    read -p "Workspace directory (default: ${DEFAULT_WORKSPACE}): " WORKSPACE_DIR
    WORKSPACE_DIR=${WORKSPACE_DIR:-$DEFAULT_WORKSPACE}
    
    # Create workspace and cache directories with correct permissions
    echo -e "${YELLOW}Creating directories with correct permissions...${NC}"
    mkdir -p "$WORKSPACE_DIR"
    mkdir -p "./cache"
    
    # Set correct ownership for cache directory
    chown -R $USER_UID:$USER_GID "./cache" 2>/dev/null || chmod -R 777 "./cache"
    
    echo -e "\n${GREEN}Configuration complete!${NC}"
}

# Function to create Dockerfile for NVIDIA container
create_dockerfile() {
    cat > Dockerfile << 'EOF'
# Use NVIDIA's official PyTorch container (has all dependencies pre-configured)
FROM nvcr.io/nvidia/pytorch:24.06-py3

# Update system packages and install required tools
RUN apt-get update && apt-get install -y \
    vim \
    curl \
    wget \
    tini \
    zip \
    unzip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install ninja for faster compilation
RUN pip install ninja

# Upgrade to latest JupyterLab and install extensions for Python file support
RUN pip install --upgrade \
    jupyterlab==4.* \
    ipywidgets \
    jupyterlab-code-formatter \
    black \
    isort \
    nbconvert \
    ipykernel

# Install other common packages
RUN pip install \
    pandas matplotlib seaborn plotly \
    scikit-learn transformers datasets \
    accelerate sentence-transformers

# Install flash-attn (should work out of the box with this container)
ENV MAX_JOBS=4
RUN pip install flash-attn --no-build-isolation

# User creation arguments
ARG USERNAME=pardo
ARG USER_UID=1012
ARG USER_GID=1013

# Create user and group
RUN groupadd -g $USER_GID $USERNAME || true && \
    useradd -m -u $USER_UID -g $USER_GID -s /bin/bash $USERNAME || true

# Add user to sudo group
RUN usermod -aG sudo $USERNAME 2>/dev/null || true

# Set environment variable for runtime
ENV USERNAME=$USERNAME

# Create directories with correct permissions
RUN mkdir -p /home/$USERNAME/workspace && \
    mkdir -p /home/$USERNAME/.cache && \
    chown -R $USERNAME:$USER_GID /home/$USERNAME

# Switch to user
USER $USERNAME
WORKDIR /home/$USERNAME/workspace

# Set up JupyterLab configuration with Python file support
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.allow_origin = '*'" >> /home/$USERNAME/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /home/$USERNAME/.jupyter/jupyter_lab_config.py

# Expose port
EXPOSE 8977

# Use tini as entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start JupyterLab with enhanced configuration
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8977 --no-browser --notebook-dir=/home/$USERNAME/workspace --allow-root"]
EOF

    echo -e "${GREEN}Dockerfile created (NVIDIA PyTorch container)${NC}"
}

# Function to create docker-compose.yml
create_docker_compose() {
    cat > docker-compose.yml << EOF
version: '3.8'

services:
  cuda-jupyter:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        USERNAME: $USERNAME
        USER_UID: $USER_UID
        USER_GID: $USER_GID
    image: cuda-jupyter
    container_name: $CONTAINER_NAME
    ports:
      - "$JUPYTER_PORT:8977"
    volumes:
      - "$WORKSPACE_DIR:/home/$USERNAME/workspace"
      - "./cache:/home/$USERNAME/.cache"
    environment:
      - JUPYTER_TOKEN=$JUPYTER_PASSWORD
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    stdin_open: true
    tty: true
    networks:
      - cuda-jupyter-network

networks:
  cuda-jupyter-network:
    driver: bridge
EOF

    echo -e "${GREEN}docker-compose.yml created${NC}"
}

# Function to create requirements.txt
create_requirements() {
    cat > requirements.txt << 'EOF'
# Core ML/DL libraries (pre-installed in NVIDIA container)
torch
torchvision
torchaudio
transformers
datasets
accelerate
flash-attn

# Data science essentials
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn

# Jupyter essentials (upgraded versions)
jupyterlab>=4.0
ipywidgets
nbconvert
ipykernel
jupyterlab-code-formatter
black
isort

# Additional useful packages
tqdm
requests
pillow
opencv-python
einops
wandb
tensorboard
sentence-transformers
EOF

    echo -e "${GREEN}requirements.txt created${NC}"
}

# Function to create test notebook
create_test_notebook() {
    mkdir -p "$WORKSPACE_DIR"
    cat > "$WORKSPACE_DIR/test_cuda_setup.ipynb" << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CUDA and Flash Attention Test\n",
    "\n",
    "This notebook tests the CUDA setup and flash_attention installation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import sys\n",
    "\n",
    "print(f\"Python version: {sys.version}\")\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"CUDA version: {torch.version.cuda}\")\n",
    "    print(f\"GPU count: {torch.cuda.device_count()}\")\n",
    "    for i in range(torch.cuda.device_count()):\n",
    "        print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test flash attention\n",
    "try:\n",
    "    import flash_attn\n",
    "    print(f\"Flash Attention version: {flash_attn.__version__}\")\n",
    "    print(\"Flash Attention successfully imported!\")\n",
    "except ImportError as e:\n",
    "    print(f\"Flash Attention import failed: {e}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test Hugging Face model download\n",
    "from transformers import AutoTokenizer\n",
    "\n",
    "try:\n",
    "    tokenizer = AutoTokenizer.from_pretrained(\"microsoft/DialoGPT-small\")\n",
    "    print(\"✓ Hugging Face cache working - model downloaded successfully!\")\n",
    "    print(f\"Cache directory: /home/pardo/.cache/huggingface\")\n",
    "except Exception as e:\n",
    "    print(f\"✗ Hugging Face cache error: {e}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simple CUDA test\n",
    "if torch.cuda.is_available():\n",
    "    device = torch.device('cuda')\n",
    "    x = torch.randn(1000, 1000, device=device)\n",
    "    y = torch.randn(1000, 1000, device=device)\n",
    "    z = torch.mm(x, y)\n",
    "    print(f\"Matrix multiplication on GPU successful!\")\n",
    "    print(f\"Result shape: {z.shape}\")\n",
    "    print(f\"Result device: {z.device}\")\n",
    "else:\n",
    "    print(\"CUDA not available, skipping GPU test\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

    # Create a test Python file
    cat > "$WORKSPACE_DIR/test_script.py" << 'EOF'
#!/usr/bin/env python3
"""
Test Python script for JupyterLab execution
"""

import torch
import numpy as np

def main():
    print("=== Python Script Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # Simple GPU computation
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        result = torch.matmul(x, y)
        print(f"GPU computation successful! Result shape: {result.shape}")
    
    # Test Hugging Face cache
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        print("✓ Hugging Face cache working!")
    except Exception as e:
        print(f"✗ Cache error: {e}")
    
    print("Python file execution works!")

if __name__ == "__main__":
    main()
EOF

    echo -e "${GREEN}Test notebook and Python script created in $WORKSPACE_DIR${NC}"
}

# Function to create startup script
create_startup_script() {
    cat > start.sh << EOF
#!/bin/bash

echo -e "${CYAN}Starting CUDA JupyterLab container (NVIDIA Edition)...${NC}"

# Ensure cache directory has correct permissions
if [ -d "./cache" ]; then
    echo "Setting cache permissions..."
    chown -R $USER_UID:$USER_GID ./cache 2>/dev/null || chmod -R 777 ./cache
fi

# Build and start the container
docker compose up --build -d

if [ \$? -eq 0 ]; then
    echo -e "\n${GREEN}Container started successfully!${NC}"
    echo -e "Access JupyterLab at: ${CYAN}http://localhost:$JUPYTER_PORT${NC}"
    echo -e "Password: ${CYAN}$JUPYTER_PASSWORD${NC}"
    echo -e "\n${YELLOW}Features available:${NC}"
    echo -e "  ✓ Latest JupyterLab 4.x with Python file support"
    echo -e "  ✓ NVIDIA PyTorch container with CUDA support"
    echo -e "  ✓ Flash-attention pre-installed"
    echo -e "  ✓ Fixed Hugging Face cache permissions"
    echo -e "  ✓ Code formatting and Python execution"
    echo -e "\n${YELLOW}Commands:${NC}"
    echo -e "  View logs: ${CYAN}docker compose logs -f${NC}"
    echo -e "  Stop: ${CYAN}docker compose down${NC}"
    echo -e "  Rebuild: ${CYAN}docker compose up --build${NC}"
else
    echo -e "\n${RED}Failed to start container. Check logs with:${NC}"
    echo -e "${CYAN}docker compose logs${NC}"
fi
EOF

    chmod +x start.sh
    echo -e "${GREEN}start.sh script created${NC}"
}

# Function to create stop script
create_stop_script() {
    cat > stop.sh << 'EOF'
#!/bin/bash

echo "Stopping CUDA JupyterLab container..."
docker compose down

echo "Container stopped."
echo "To start again, run: ./start.sh"
EOF

    chmod +x stop.sh
    echo -e "${GREEN}stop.sh script created${NC}"
}

# Function to show summary
show_summary() {
    echo -e "\n${CYAN}======================================================${NC}"
    echo -e "${CYAN}     CUDA JupyterLab Setup Complete (NVIDIA)${NC}"
    echo -e "${CYAN}======================================================${NC}"
    echo -e "${GREEN}Configuration:${NC}"
    echo -e "  Container: ${CYAN}$CONTAINER_NAME${NC}"
    echo -e "  Base Image: ${CYAN}nvcr.io/nvidia/pytorch:24.06-py3${NC}"
    echo -e "  Port: ${CYAN}$JUPYTER_PORT${NC}"
    echo -e "  Username: ${CYAN}$USERNAME${NC}"
    echo -e "  UID/GID: ${CYAN}$USER_UID/$USER_GID${NC}"
    echo -e "  Workspace: ${CYAN}$WORKSPACE_DIR${NC}"
    echo -e "  Cache: ${CYAN}./cache${NC}"
    echo -e "  Password: ${CYAN}$JUPYTER_PASSWORD${NC}"
    echo -e "\n${YELLOW}Features:${NC}"
    echo -e "  ✓ Latest JupyterLab 4.x"
    echo -e "  ✓ Python file editing and execution"
    echo -e "  ✓ Code formatting (Black, isort)"
    echo -e "  ✓ Flash-attention support"
    echo -e "  ✓ CUDA 12.4+ ready"
    echo -e "  ✓ PyTorch pre-installed"
    echo -e "  ✓ Fixed Hugging Face cache permissions"
    echo -e "\n${YELLOW}Files created:${NC}"
    echo -e "  - Dockerfile (NVIDIA PyTorch base)"
    echo -e "  - docker-compose.yml (with local cache)"
    echo -e "  - requirements.txt"
    echo -e "  - start.sh & stop.sh scripts"
    echo -e "  - test_cuda_setup.ipynb"
    echo -e "  - test_script.py"
    echo -e "\n${GREEN}To start:${NC} ${CYAN}./start.sh${NC}"
    echo -e "${GREEN}To stop:${NC} ${CYAN}./stop.sh${NC}"
    echo -e "${CYAN}======================================================${NC}"
}

# Main execution
main() {
    echo -e "${CYAN}======================================================${NC}"
    echo -e "${CYAN}      CUDA JupyterLab Setup (NVIDIA Container)${NC}"
    echo -e "${CYAN}======================================================${NC}"
    
    check_prerequisites
    get_configuration
    
    echo -e "\n${YELLOW}Creating configuration files...${NC}"
    create_dockerfile
    create_docker_compose
    create_requirements
    create_test_notebook
    create_startup_script
    create_stop_script
    
    show_summary
}

# Run main function
main "$@"