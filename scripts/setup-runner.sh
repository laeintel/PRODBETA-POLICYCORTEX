#!/bin/bash
# Setup script for self-hosted GitHub Actions runners

set -e

echo "🚀 Setting up self-hosted runner environment..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"

# Install Docker
install_docker() {
    if command -v docker &> /dev/null; then
        echo "✅ Docker already installed"
        docker --version
        return
    fi
    
    echo "📦 Installing Docker..."
    
    if [ "$OS" == "linux" ]; then
        # Install Docker on Linux
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        
        # Add current user to docker group
        sudo usermod -aG docker $USER
        
        # Start Docker service
        sudo systemctl start docker
        sudo systemctl enable docker
        
    elif [ "$OS" == "macos" ]; then
        # Install Docker on macOS
        if command -v brew &> /dev/null; then
            brew install --cask docker
        else
            echo "Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
            exit 1
        fi
        
    elif [ "$OS" == "windows" ]; then
        # Install Docker on Windows
        echo "Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
        echo "Or use WSL2 with Docker"
        exit 1
    fi
    
    echo "✅ Docker installed successfully"
}

# Install Azure CLI
install_azure_cli() {
    if command -v az &> /dev/null; then
        echo "✅ Azure CLI already installed"
        az --version
        return
    fi
    
    echo "📦 Installing Azure CLI..."
    
    if [ "$OS" == "linux" ]; then
        # Install Azure CLI on Linux
        curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
        
    elif [ "$OS" == "macos" ]; then
        # Install Azure CLI on macOS
        if command -v brew &> /dev/null; then
            brew install azure-cli
        else
            echo "Please install Homebrew first: https://brew.sh"
            exit 1
        fi
        
    elif [ "$OS" == "windows" ]; then
        # Install Azure CLI on Windows
        echo "Download from: https://aka.ms/installazurecliwindows"
        exit 1
    fi
    
    echo "✅ Azure CLI installed successfully"
}

# Install Node.js
install_nodejs() {
    if command -v node &> /dev/null; then
        echo "✅ Node.js already installed"
        node --version
        return
    fi
    
    echo "📦 Installing Node.js..."
    
    # Install via Node Version Manager (nvm)
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    source ~/.bashrc
    nvm install 20
    nvm use 20
    
    echo "✅ Node.js installed successfully"
}

# Install Rust
install_rust() {
    if command -v cargo &> /dev/null; then
        echo "✅ Rust already installed"
        cargo --version
        return
    fi
    
    echo "📦 Installing Rust..."
    
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    
    echo "✅ Rust installed successfully"
}

# Install Terraform
install_terraform() {
    if command -v terraform &> /dev/null; then
        echo "✅ Terraform already installed"
        terraform --version
        return
    fi
    
    echo "📦 Installing Terraform..."
    
    # Install Terraform
    wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
    sudo apt update && sudo apt install terraform
    
    echo "✅ Terraform installed successfully"
}

# Main installation
main() {
    echo "Installing required tools for PolicyCortex CI/CD..."
    
    install_docker
    install_azure_cli
    install_nodejs
    install_rust
    install_terraform
    
    echo ""
    echo "✅ All tools installed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Log out and log back in (for Docker group changes)"
    echo "2. Run 'az login' to authenticate with Azure"
    echo "3. Configure GitHub Actions runner"
    echo ""
    echo "To verify installation:"
    echo "  docker --version"
    echo "  az --version"
    echo "  node --version"
    echo "  cargo --version"
    echo "  terraform --version"
}

# Run main function
main