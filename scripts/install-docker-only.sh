#!/bin/bash
# Simple Docker installation script for self-hosted runners

set -e

echo "🐳 Docker Installation Script"
echo "============================="

# Check if Docker is already installed
if command -v docker &> /dev/null; then
    echo "✅ Docker is already installed:"
    docker --version
    
    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        echo "✅ Docker daemon is running"
    else
        echo "⚠️ Docker is installed but daemon is not running"
        echo "Try: sudo systemctl start docker"
    fi
    exit 0
fi

echo "📦 Installing Docker..."

# Detect OS and install accordingly
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"
    
    # Method 1: Official Docker script (most reliable)
    echo "Using official Docker installation script..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    
    # Add current user to docker group
    echo "Adding $USER to docker group..."
    sudo usermod -aG docker $USER || true
    
    # Start Docker
    echo "Starting Docker service..."
    sudo systemctl start docker || sudo service docker start || true
    sudo systemctl enable docker || true
    
    echo ""
    echo "✅ Docker installed successfully!"
    echo ""
    echo "⚠️ IMPORTANT: You need to log out and log back in for group changes to take effect"
    echo "Or run: newgrp docker"
    echo ""
    echo "To verify: docker run hello-world"
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    echo ""
    echo "Please install Docker Desktop from:"
    echo "https://www.docker.com/products/docker-desktop"
    echo ""
    echo "Or using Homebrew:"
    echo "brew install --cask docker"
    
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows"
    echo ""
    echo "Please install Docker Desktop from:"
    echo "https://www.docker.com/products/docker-desktop"
    echo ""
    echo "Make sure to enable WSL2 backend for best performance"
    
else
    echo "⚠️ Unknown operating system: $OSTYPE"
    echo "Please install Docker manually from https://docs.docker.com/get-docker/"
fi