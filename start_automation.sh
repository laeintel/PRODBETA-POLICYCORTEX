#!/bin/bash

# PolicyCortex Claude Code Automation Startup Script
# Automates development using claude-code with instruction files

set -e

echo "🚀 PolicyCortex Claude Code Automation"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if claude-code is available
if ! command -v claude-code &> /dev/null; then
    print_error "claude-code is not installed or not in PATH"
    print_error "Please install Claude Code (Cursor) first"
    exit 1
fi

print_success "Claude Code found"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

print_success "Python 3 found"

# Create project structure if it doesn't exist
PROJECT_ROOT="/workspace/policycortex"
if [ ! -d "$PROJECT_ROOT" ]; then
    print_status "Creating project structure..."
    mkdir -p "$PROJECT_ROOT"/{core/src,frontend/app,ml,templates,tests,docs,scripts,config}
    
    # Initialize git repository
    cd "$PROJECT_ROOT"
    git init
    git checkout -b development
    
    # Create initial files
    echo "# PolicyCortex - AI-Powered Cloud Governance Platform" > README.md
    echo "target/" > .gitignore
    echo "node_modules/" >> .gitignore
    echo "*.log" >> .gitignore
    echo ".env" >> .gitignore
    echo "dist/" >> .gitignore
    
    git add .
    git commit -m "Initial commit - PolicyCortex automation setup"
    
    print_success "Project structure created"
fi

# Make automation script executable
chmod +x claude_automation_system.py

# Create automation configuration
print_status "Creating automation configuration..."

cat > automation_config.yaml << 'EOF'
# PolicyCortex Automation Configuration

# Automation settings
cycle_interval: 1800  # 30 minutes between cycles
max_concurrent_tasks: 1
quality_threshold: 70
auto_commit: true

# Project settings
project_root: "/workspace/policycortex"

# Task queue (will execute in order)
task_queue:
  - "day1-3_remediation.txt"
  - "day4-6_ml_predictions.txt"
  - "day7-9_correlation.txt"
  - "day10-12_nlp_interface.txt"
  - "day13-14_integration.txt"

# Notification settings (optional)
notification_webhook: null  # Set to Slack webhook URL if desired

# Git settings
git_auto_commit: true
git_branch: "development"

# Logging
log_level: "INFO"
log_retention_days: 30

# Quality gates
quality_gates:
  minimum_test_coverage: 80
  maximum_complexity: 10
  security_scan: true

# Development phases
phases:
  phase1:
    name: "Foundation & Core AI"
    duration_weeks: 12
    tasks: ["remediation", "ml_predictions", "correlation"]
  
  phase2:
    name: "Domain Implementation"
    duration_weeks: 12
    tasks: ["nlp_interface", "integration", "testing"]
  
  phase3:
    name: "Scale & Optimize"
    duration_weeks: 12
    tasks: ["production", "monitoring", "deployment"]
EOF

print_success "Configuration created"

# Function to run automation modes
run_generate_only() {
    print_status "Generating instruction files only..."
    python3 claude_automation_system.py --generate --project-root "$PROJECT_ROOT"
    print_success "Instruction files generated in ./claude_instructions/"
    
    echo ""
    echo "📋 Generated Files:"
    ls -la claude_instructions/
    echo ""
    echo "You can now run individual files with:"
    echo "  claude-code --yes < claude_instructions/day1-3_remediation.txt"
}

run_single_cycle() {
    print_status "Running single automation cycle..."
    python3 claude_automation_system.py --run-once --project-root "$PROJECT_ROOT"
    print_success "Single cycle completed"
}

run_continuous() {
    print_status "Starting continuous automation..."
    print_warning "This will run indefinitely. Press Ctrl+C to stop."
    
    # Create systemd service for production use
    if command -v systemctl &> /dev/null; then
        print_status "Creating systemd service..."
        
        cat > /tmp/policycortex-automation.service << EOF
[Unit]
Description=PolicyCortex Claude Code Automation
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $(pwd)/claude_automation_system.py --continuous --project-root $PROJECT_ROOT
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        print_status "To install as system service:"
        echo "  sudo cp /tmp/policycortex-automation.service /etc/systemd/system/"
        echo "  sudo systemctl enable policycortex-automation"
        echo "  sudo systemctl start policycortex-automation"
        echo ""
    fi
    
    # Run directly
    python3 claude_automation_system.py --continuous --project-root "$PROJECT_ROOT"
}

run_interactive_mode() {
    print_status "Running in interactive mode..."
    
    while true; do
        echo ""
        echo "🤖 PolicyCortex Automation Menu"
        echo "==============================="
        echo "1. Generate instruction files only"
        echo "2. Run single automation cycle"
        echo "3. Start continuous automation"
        echo "4. Execute specific instruction file"
        echo "5. View automation logs"
        echo "6. Edit configuration"
        echo "7. Exit"
        echo ""
        read -p "Select option (1-7): " choice
        
        case $choice in
            1)
                run_generate_only
                ;;
            2)
                run_single_cycle
                ;;
            3)
                run_continuous
                ;;
            4)
                echo ""
                echo "Available instruction files:"
                ls -1 claude_instructions/ 2>/dev/null || echo "No files generated yet"
                echo ""
                read -p "Enter filename: " filename
                if [ -f "claude_instructions/$filename" ]; then
                    print_status "Executing $filename..."
                    cd "$PROJECT_ROOT"
                    claude-code --yes < "../claude_instructions/$filename"
                    print_success "Execution completed"
                else
                    print_error "File not found: $filename"
                fi
                ;;
            5)
                if [ -f "automation_logs/automation.log" ]; then
                    tail -50 automation_logs/automation.log
                else
                    print_warning "No logs found yet"
                fi
                ;;
            6)
                ${EDITOR:-nano} automation_config.yaml
                ;;
            7)
                print_success "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
    done
}

# Parse command line arguments
case "${1:-interactive}" in
    "generate")
        run_generate_only
        ;;
    "once")
        run_single_cycle
        ;;
    "continuous")
        run_continuous
        ;;
    "interactive")
        run_interactive_mode
        ;;
    *)
        echo "Usage: $0 [generate|once|continuous|interactive]"
        echo ""
        echo "Options:"
        echo "  generate     - Generate instruction files only"
        echo "  once         - Run one automation cycle"
        echo "  continuous   - Run continuous automation (background)"
        echo "  interactive  - Interactive menu (default)"
        echo ""
        echo "Examples:"
        echo "  $0 generate              # Generate files only"
        echo "  $0 once                  # Run once and exit"
        echo "  $0 continuous            # Run forever"
        echo "  $0                       # Interactive menu"
        exit 1
        ;;
esac

