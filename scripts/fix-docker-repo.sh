#!/bin/bash

# Fix Docker Repository Configuration Script
# This script fixes corrupted Docker repository entries that cause APT errors

set -e

echo "🔧 Docker Repository Fix Script"
echo "==============================="
echo

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root. Run as regular user with sudo access."
   exit 1
fi

# Check for sudo access
if ! sudo -v; then
    echo "❌ This script requires sudo access"
    exit 1
fi

echo "🔍 Checking current Docker repository status..."

# Check for corrupted entries
CORRUPTED=false
if grep -r "download\.docker\.com.*|" /etc/apt/sources.list* 2>/dev/null; then
    echo "❌ Found corrupted Docker repository entries with pipe characters"
    CORRUPTED=true
else
    echo "✅ No corrupted pipe entries found"
fi

# Try to update package list
UPDATE_FAILED=false
if ! sudo apt-get update >/dev/null 2>&1; then
    echo "❌ apt-get update failed"
    UPDATE_FAILED=true
else
    echo "✅ apt-get update succeeded"
fi

# Check if Docker is installed and working
DOCKER_BROKEN=false
if ! command -v docker >/dev/null 2>&1; then
    echo "⚠️ Docker is not installed"
    DOCKER_BROKEN=true
elif ! sudo docker info >/dev/null 2>&1; then
    echo "⚠️ Docker daemon is not running or accessible"
    DOCKER_BROKEN=true
else
    echo "✅ Docker appears to be working"
fi

# Determine if fix is needed
if [[ "$CORRUPTED" == "true" ]] || [[ "$UPDATE_FAILED" == "true" ]] || [[ "$DOCKER_BROKEN" == "true" ]]; then
    echo
    echo "🔧 Docker repository fix is needed. Proceeding..."
    echo
else
    echo
    echo "✅ Docker repository appears to be healthy. No fix needed."
    echo "Use --force flag to fix anyway."
    if [[ "$1" != "--force" ]]; then
        exit 0
    fi
fi

echo "🧹 Step 1: Cleaning up corrupted Docker repository entries..."

# Remove all Docker-related repository files
sudo rm -f /etc/apt/sources.list.d/docker*.list || true
sudo rm -f /etc/apt/keyrings/docker.gpg || true
sudo rm -f /etc/apt/trusted.gpg.d/docker.gpg || true

# Remove any corrupted entries from main sources.list
sudo sed -i '/download\.docker\.com.*|/d' /etc/apt/sources.list || true

# Clean all sources.list.d files
for file in /etc/apt/sources.list.d/*.list; do
    if [ -f "$file" ]; then
        sudo sed -i '/download\.docker\.com.*|/d' "$file" || true
    fi
done

echo "✅ Corrupted entries removed"

echo
echo "🧹 Step 2: Cleaning package cache..."

# Clean package cache
sudo apt-get clean || true
sudo rm -rf /var/lib/apt/lists/* || true

echo "✅ Package cache cleaned"

echo
echo "📦 Step 3: Updating package index..."

# Update package index
if sudo apt-get update; then
    echo "✅ Package index updated successfully"
else
    echo "❌ Package index update failed"
    exit 1
fi

echo
echo "🐳 Step 4: Installing/Reinstalling Docker..."

# Remove existing Docker installation
sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Install Docker using official script
echo "📦 Installing Docker using official script..."
curl -fsSL https://get.docker.com -o get-docker.sh

if sudo sh get-docker.sh; then
    echo "✅ Docker installation completed"
else
    echo "❌ Docker installation failed"
    rm -f get-docker.sh
    exit 1
fi

rm -f get-docker.sh

echo
echo "👥 Step 5: Configuring user permissions..."

# Add users to docker group
sudo usermod -aG docker $USER || true
sudo usermod -aG docker runner || true
sudo usermod -aG docker github || true

echo "✅ User permissions configured"

echo
echo "🚀 Step 6: Starting Docker service..."

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

echo "✅ Docker service started and enabled"

echo
echo "🔍 Step 7: Verifying Docker installation..."

# Wait a moment for Docker to start
sleep 3

# Check Docker version
if docker --version; then
    echo "✅ Docker version check passed"
else
    echo "❌ Docker version check failed"
    exit 1
fi

# Check Docker daemon
if sudo docker info >/dev/null 2>&1; then
    echo "✅ Docker daemon is running"
else
    echo "❌ Docker daemon is not running"
    echo "🔄 Attempting to restart Docker daemon..."
    sudo systemctl restart docker
    sleep 5
    if sudo docker info >/dev/null 2>&1; then
        echo "✅ Docker daemon started successfully"
    else
        echo "❌ Failed to start Docker daemon"
        exit 1
    fi
fi

# Test Docker functionality
echo "🧪 Testing Docker functionality..."
if sudo docker run --rm hello-world >/dev/null 2>&1; then
    echo "✅ Docker test container ran successfully"
else
    echo "❌ Docker test container failed"
    exit 1
fi

echo
echo "🎉 SUCCESS: Docker repository has been fixed and Docker is working correctly!"
echo
echo "⚠️ IMPORTANT: You may need to log out and log back in (or restart the runner)"
echo "   for Docker group permissions to take effect for your user account."
echo
echo "📋 Summary of actions taken:"
echo "   ✅ Removed corrupted Docker repository entries"
echo "   ✅ Cleaned package cache and sources"
echo "   ✅ Reinstalled Docker using official script"
echo "   ✅ Configured user permissions"
echo "   ✅ Verified Docker functionality"
echo
echo "🚀 Your CI/CD pipelines should now work correctly!"