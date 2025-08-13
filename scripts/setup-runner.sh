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
        # Safer Azure CLI installation
        if command -v apt-get &> /dev/null; then
            # For Debian/Ubuntu
            sudo apt-get update || true
            sudo apt-get install -y ca-certificates curl apt-transport-https lsb-release gnupg || true
            
            # Microsoft signing key
            curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null
            
            # Add repository
            AZ_REPO=$(lsb_release -cs 2>/dev/null || echo "focal")
            echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | sudo tee /etc/apt/sources.list.d/azure-cli.list
            
            # Install
            sudo apt-get update || true
            sudo apt-get install -y azure-cli || true
        else
            # Direct installation via pip
            echo "Installing Azure CLI via Python pip..."
            pip3 install --user azure-cli || pip install --user azure-cli
        fi
        
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
    
    if [ "$OS" == "linux" ]; then
        # Check if we have apt-get
        if command -v apt-get &> /dev/null; then
            # Safer installation method
            sudo apt-get update || true
            sudo apt-get install -y wget unzip || true
            
            # Download Terraform directly
            TERRAFORM_VERSION="1.6.0"
            wget -q "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
            unzip -o "terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
            sudo mv terraform /usr/local/bin/
            rm "terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
        else
            echo "apt-get not found, trying direct download..."
            # Direct binary download for non-Debian systems
            TERRAFORM_VERSION="1.6.0"
            curl -LO "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
            unzip -o "terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
            sudo mv terraform /usr/local/bin/ || mv terraform ~/bin/
            rm "terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
        fi
    elif [ "$OS" == "macos" ]; then
        if command -v brew &> /dev/null; then
            brew tap hashicorp/tap
            brew install hashicorp/tap/terraform
        else
            # Direct download for macOS
            TERRAFORM_VERSION="1.6.0"
            curl -LO "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_darwin_amd64.zip"
            unzip -o "terraform_${TERRAFORM_VERSION}_darwin_amd64.zip"
            sudo mv terraform /usr/local/bin/
            rm "terraform_${TERRAFORM_VERSION}_darwin_amd64.zip"
        fi
    fi
    
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