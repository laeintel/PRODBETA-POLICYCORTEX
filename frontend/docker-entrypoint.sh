#!/bin/sh

# Docker entrypoint script for PolicyCortex Frontend

set -e

# Function to substitute environment variables in JavaScript files
substitute_env_vars() {
    local file="$1"
    local temp_file=$(mktemp)
    
    # List of environment variables to substitute
    local env_vars="VITE_API_BASE_URL VITE_WS_URL VITE_AZURE_CLIENT_ID VITE_AZURE_TENANT_ID VITE_AZURE_REDIRECT_URI VITE_APP_VERSION"
    
    # Copy original file to temp
    cp "$file" "$temp_file"
    
    # Substitute each environment variable
    for var in $env_vars; do
        local value=$(eval echo \$$var)
        if [ -n "$value" ]; then
            # Escape special characters for sed
            local escaped_value=$(echo "$value" | sed 's/[[\.*^$()+?{|]/\\&/g')
            sed -i "s|__${var}__|${escaped_value}|g" "$temp_file"
        fi
    done
    
    # Replace original file
    mv "$temp_file" "$file"
}

# Function to generate environment configuration
generate_env_config() {
    local env_file="/usr/share/nginx/html/env-config.js"
    
    cat > "$env_file" << EOF
window.ENV = {
  API_BASE_URL: '${VITE_API_BASE_URL:-http://localhost:8000/api}',
  WS_URL: '${VITE_WS_URL:-ws://localhost:8000/ws}',
  AZURE_CLIENT_ID: '${VITE_AZURE_CLIENT_ID:-}',
  AZURE_TENANT_ID: '${VITE_AZURE_TENANT_ID:-}',
  AZURE_REDIRECT_URI: '${VITE_AZURE_REDIRECT_URI:-}',
  APP_VERSION: '${VITE_APP_VERSION:-1.0.0}',
  NODE_ENV: '${NODE_ENV:-production}',
  ENABLE_ANALYTICS: '${VITE_ENABLE_ANALYTICS:-false}',
  ENABLE_NOTIFICATIONS: '${VITE_ENABLE_NOTIFICATIONS:-true}',
  ENABLE_WEBSOCKET: '${VITE_ENABLE_WEBSOCKET:-true}',
  ENABLE_PWA: '${VITE_ENABLE_PWA:-true}',
  ENABLE_DARK_MODE: '${VITE_ENABLE_DARK_MODE:-true}',
  LOG_LEVEL: '${VITE_LOG_LEVEL:-info}',
  ENABLE_DEBUG: '${VITE_ENABLE_DEBUG:-false}',
};
EOF
    
    echo "Environment configuration generated at $env_file"
}

# Function to validate required environment variables
validate_env() {
    local missing_vars=""
    
    # Check required environment variables
    if [ -z "$VITE_AZURE_CLIENT_ID" ]; then
        missing_vars="$missing_vars VITE_AZURE_CLIENT_ID"
    fi
    
    if [ -z "$VITE_AZURE_TENANT_ID" ]; then
        missing_vars="$missing_vars VITE_AZURE_TENANT_ID"
    fi
    
    if [ -n "$missing_vars" ]; then
        echo "ERROR: Missing required environment variables:$missing_vars"
        echo "Please set these environment variables before starting the container."
        exit 1
    fi
}

# Function to update nginx configuration
update_nginx_config() {
    local nginx_conf="/etc/nginx/nginx.conf"
    
    # If API_BASE_URL is set, update proxy configuration
    if [ -n "$VITE_API_BASE_URL" ]; then
        # Extract host from API_BASE_URL
        local api_host=$(echo "$VITE_API_BASE_URL" | sed 's|https\?://||' | sed 's|/.*||')
        
        # Update nginx configuration to proxy to the correct backend
        sed -i "s|proxy_pass http://backend:8000/api/;|proxy_pass ${VITE_API_BASE_URL}/;|g" "$nginx_conf"
        
        echo "Updated nginx configuration to proxy to $VITE_API_BASE_URL"
    fi
    
    # If WS_URL is set, update WebSocket proxy configuration
    if [ -n "$VITE_WS_URL" ]; then
        # Convert WebSocket URL to HTTP for proxy
        local ws_http_url=$(echo "$VITE_WS_URL" | sed 's|^ws://|http://|' | sed 's|^wss://|https://|')
        
        # Update nginx configuration
        sed -i "s|proxy_pass http://backend:8000/ws/;|proxy_pass ${ws_http_url}/;|g" "$nginx_conf"
        
        echo "Updated nginx configuration to proxy WebSocket to $ws_http_url"
    fi
}

# Function to set up health check
setup_health_check() {
    local health_file="/usr/share/nginx/html/health"
    
    cat > "$health_file" << EOF
{
  "status": "healthy",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "version": "${VITE_APP_VERSION:-1.0.0}",
  "environment": "${NODE_ENV:-production}"
}
EOF
    
    echo "Health check endpoint configured"
}

# Function to display startup info
display_startup_info() {
    echo "=================================================="
    echo "PolicyCortex Frontend Container Starting"
    echo "=================================================="
    echo "Version: ${VITE_APP_VERSION:-1.0.0}"
    echo "Environment: ${NODE_ENV:-production}"
    echo "API Base URL: ${VITE_API_BASE_URL:-http://localhost:8000/api}"
    echo "WebSocket URL: ${VITE_WS_URL:-ws://localhost:8000/ws}"
    echo "Azure Client ID: ${VITE_AZURE_CLIENT_ID:-not set}"
    echo "Azure Tenant ID: ${VITE_AZURE_TENANT_ID:-not set}"
    echo "Features:"
    echo "  - Analytics: ${VITE_ENABLE_ANALYTICS:-false}"
    echo "  - Notifications: ${VITE_ENABLE_NOTIFICATIONS:-true}"
    echo "  - WebSocket: ${VITE_ENABLE_WEBSOCKET:-true}"
    echo "  - PWA: ${VITE_ENABLE_PWA:-true}"
    echo "  - Dark Mode: ${VITE_ENABLE_DARK_MODE:-true}"
    echo "  - Debug: ${VITE_ENABLE_DEBUG:-false}"
    echo "=================================================="
}

# Main execution
main() {
    echo "Starting PolicyCortex Frontend initialization..."
    
    # Display startup information
    display_startup_info
    
    # Validate environment variables
    validate_env
    
    # Generate environment configuration
    generate_env_config
    
    # Update nginx configuration
    update_nginx_config
    
    # Set up health check
    setup_health_check
    
    # Test nginx configuration
    echo "Testing nginx configuration..."
    nginx -t
    
    if [ $? -eq 0 ]; then
        echo "Nginx configuration is valid"
    else
        echo "ERROR: Nginx configuration is invalid"
        exit 1
    fi
    
    echo "Initialization complete. Starting nginx..."
    
    # Execute the command passed to the script
    exec "$@"
}

# Run main function
main "$@"