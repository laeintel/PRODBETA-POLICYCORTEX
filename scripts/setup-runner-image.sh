#!/bin/bash
set -e

echo "Starting Azure Self-Hosted Runner Image Setup for Ubuntu 24.04"
echo "============================================================="

# Pre-flight: ensure Docker APT source is valid and never contains stray characters
repair_docker_apt_source() {
    echo "Pre-flight: ensuring Docker APT source is valid..."
    UBUNTU_CODENAME=$(. /etc/os-release 2>/dev/null && echo "$VERSION_CODENAME")
    if [ -z "$UBUNTU_CODENAME" ]; then
        UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || true)
    fi
    sudo install -m 0755 -d /etc/apt/keyrings
    if [ ! -f /etc/apt/keyrings/docker.asc ]; then
        sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        sudo chmod a+r /etc/apt/keyrings/docker.asc
    fi
    if [ -n "$UBUNTU_CODENAME" ]; then
        # Write as a single line to avoid accidental pipes/newlines in sources.list
        printf "deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n" \
          "$(dpkg --print-architecture)" "$UBUNTU_CODENAME" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    fi
}

repair_docker_apt_source

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential utilities
echo "Installing essential utilities..."
sudo apt install -y \
    curl \
    wget \
    git \
    unzip \
    zip \
    jq \
    tree \
    htop \
    vim \
    nano \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    net-tools \
    iputils-ping

# Install PowerShell Core
echo "Installing PowerShell Core..."
# Download the Microsoft repository keys
wget -q "https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb"
# Register the Microsoft repository keys
sudo dpkg -i packages-microsoft-prod.deb
# Delete the the Microsoft repository keys file
rm packages-microsoft-prod.deb
# Update the list of packages after we added packages.microsoft.com
sudo apt update
# Install PowerShell
sudo apt install -y powershell

# Install Docker
echo "Installing Docker..."
# Add Docker's official GPG key
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add (or refresh) the Docker repository to Apt sources using a single line
UBUNTU_CODENAME=$(. /etc/os-release 2>/dev/null && echo "$VERSION_CODENAME")
printf "deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n" \
  "$(dpkg --print-architecture)" "$UBUNTU_CODENAME" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update

# Install Docker packages
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Install Docker Compose standalone
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Node.js (via NodeSource repository for latest LTS)
echo "Installing Node.js and npm..."
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install global npm packages
echo "Installing global npm packages..."
sudo npm install -g \
    yarn \
    pnpm \
    pm2 \
    typescript \
    ts-node \
    eslint \
    prettier \
    jest \
    @angular/cli \
    @vue/cli \
    create-react-app \
    next \
    vite \
    webpack \
    webpack-cli \
    nodemon

# Install Python and pip
echo "Installing Python 3 and essential packages..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev

# Install Python packages
echo "Installing Python packages..."
pip3 install --break-system-packages \
    pipenv \
    poetry \
    virtualenv \
    black \
    flake8 \
    pylint \
    pytest \
    mypy \
    jupyter \
    pandas \
    numpy \
    requests \
    django \
    flask \
    fastapi \
    uvicorn \
    sqlalchemy \
    boto3 \
    azure-cli \
    ansible

# Install .NET SDK
echo "Installing .NET SDK..."
sudo apt-get install -y dotnet-sdk-8.0

# Install Go
echo "Installing Go..."
GO_VERSION="1.22.0"
wget https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz
rm go${GO_VERSION}.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' | sudo tee -a /etc/profile

# Install Java (OpenJDK)
echo "Installing Java..."
sudo apt install -y openjdk-17-jdk openjdk-11-jdk maven gradle

# Install Ruby
echo "Installing Ruby..."
sudo apt install -y ruby-full ruby-bundler

# Install PHP
echo "Installing PHP..."
sudo apt install -y \
    php \
    php-cli \
    php-common \
    php-mysql \
    php-zip \
    php-gd \
    php-mbstring \
    php-curl \
    php-xml \
    php-bcmath \
    composer

# Install Terraform
echo "Installing Terraform..."
wget -O- https://apt.releases.hashicorp.com/gpg | \
    gpg --dearmor | \
    sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
    https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update
sudo apt install -y terraform

# Install kubectl
echo "Installing kubectl..."
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl

# Install Helm
echo "Installing Helm..."
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Azure CLI
echo "Installing Azure CLI..."
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install AWS CLI
echo "Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Install GitHub CLI
echo "Installing GitHub CLI..."
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
&& sudo mkdir -p -m 755 /etc/apt/keyrings \
&& wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

# Install database clients
echo "Installing database clients..."
sudo apt install -y \
    mysql-client \
    postgresql-client \
    redis-tools \
    sqlite3

# Install additional tools
echo "Installing additional development tools..."
sudo apt install -y \
    make \
    cmake \
    gcc \
    g++ \
    clang \
    llvm \
    ninja-build \
    meson \
    autoconf \
    automake \
    libtool \
    pkg-config

# Clean up
echo "Cleaning up..."
sudo apt autoremove -y
sudo apt autoclean

# Create runner user (if needed for Azure DevOps)
echo "Creating azdevops user..."
sudo useradd -m -s /bin/bash azdevops || true
sudo usermod -aG docker azdevops || true

# Set up environment variables
echo "Setting up environment variables..."
cat << 'EOF' | sudo tee /etc/profile.d/runner-tools.sh
# Go
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# Java
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin

# Python
export PATH=$PATH:$HOME/.local/bin

# .NET
export DOTNET_ROOT=/usr/share/dotnet
export PATH=$PATH:$DOTNET_ROOT
EOF

echo "============================================================="
echo "Installation completed successfully!"
echo "============================================================="
echo ""
echo "Installed tools:"
echo "- PowerShell Core"
echo "- Docker & Docker Compose"
echo "- Node.js & npm with global packages"
echo "- Python 3 with essential packages"
echo "- .NET SDK 8.0"
echo "- Go"
echo "- Java (OpenJDK 11 & 17)"
echo "- Ruby"
echo "- PHP & Composer"
echo "- Terraform"
echo "- kubectl & Helm"
echo "- Azure CLI"
echo "- AWS CLI"
echo "- GitHub CLI"
echo "- Database clients"
echo "- Build tools"
echo ""
echo "Next steps:"
echo "1. Reboot the VM: sudo reboot"
echo "2. After reboot, generalize the VM for image capture"
echo "3. Deallocate and capture the image"