#!/usr/bin/env python3
"""
Fix linting errors in the PolicyCortex codebase.
"""

import os
import subprocess
import sys
from pathlib import Path

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
NC = '\033[0m'  # No Color


def print_colored(message, color=NC):
    """Print colored message."""
    print(f"{color}{message}{NC}")


def fix_python_files():
    """Fix Python linting errors."""
    print_colored("\nFixing Python linting errors...", CYAN)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    backend_path = project_root / "backend"
    
    if not backend_path.exists():
        print_colored("Backend directory not found", RED)
        return
    
    # Install tools
    print_colored("Installing linting tools...", YELLOW)
    subprocess.run([sys.executable, "-m", "pip", "install", "autopep8", "black", "isort", "flake8", "--quiet"])
    
    # Run autopep8 to fix whitespace issues
    print_colored("Running autopep8 to fix whitespace issues...", YELLOW)
    for py_file in backend_path.rglob("*.py"):
        if "venv" not in str(py_file) and "__pycache__" not in str(py_file):
            subprocess.run(["autopep8", "--in-place", "--aggressive", "--aggressive", str(py_file)], 
                         capture_output=True)
    
    # Specifically fix the sentiment_analyzer.py file
    sentiment_file = backend_path / "services" / "ai_engine" / "services" / "sentiment_analyzer.py"
    if sentiment_file.exists():
        print_colored(f"Fixing {sentiment_file.name}...", YELLOW)
        
        # Read the file
        with open(sentiment_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove trailing whitespace from each line
        lines = content.splitlines()
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Remove excessive blank lines
        final_lines = []
        blank_count = 0
        for line in cleaned_lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    final_lines.append(line)
            else:
                blank_count = 0
                final_lines.append(line)
        
        # Ensure file ends with newline
        content = '\n'.join(final_lines)
        if not content.endswith('\n'):
            content += '\n'
        
        # Write back
        with open(sentiment_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # Run black for formatting
    print_colored("Running black for code formatting...", YELLOW)
    subprocess.run(["black", str(backend_path), "--line-length", "100", "--quiet"], 
                   capture_output=True)
    
    # Run isort for import sorting
    print_colored("Running isort for import sorting...", YELLOW)
    subprocess.run(["isort", str(backend_path), "--profile", "black", "--line-length", "100", "--quiet"], 
                   capture_output=True)
    
    # Check remaining issues
    print_colored("\nChecking remaining issues with flake8...", YELLOW)
    result = subprocess.run(
        ["flake8", str(backend_path), "--max-line-length=100", 
         "--exclude=venv,__pycache__,migrations", "--count"],
        capture_output=True, text=True
    )
    
    if result.stdout.strip() == "0" or not result.stdout.strip():
        print_colored("✓ Python: No linting errors found", GREEN)
    else:
        # Get detailed errors
        detailed = subprocess.run(
            ["flake8", str(backend_path), "--max-line-length=100", 
             "--exclude=venv,__pycache__,migrations"],
            capture_output=True, text=True
        )
        
        if detailed.stdout:
            print_colored(f"Remaining Python linting issues:\n{detailed.stdout}", YELLOW)
            
            # Try to fix remaining issues
            for line in detailed.stdout.splitlines():
                if "W293 blank line contains whitespace" in line or "W292 no newline at end of file" in line:
                    # Extract filename
                    parts = line.split(":")
                    if len(parts) >= 2:
                        filename = parts[0]
                        fix_whitespace_issues(filename)


def fix_whitespace_issues(filename):
    """Fix specific whitespace issues in a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into lines and clean
        lines = content.splitlines()
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            cleaned_line = line.rstrip()
            cleaned_lines.append(cleaned_line)
        
        # Join lines and ensure newline at end
        content = '\n'.join(cleaned_lines)
        if content and not content.endswith('\n'):
            content += '\n'
        
        # Write back
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print_colored(f"  Fixed whitespace in {os.path.basename(filename)}", GREEN)
        
    except Exception as e:
        print_colored(f"  Error fixing {filename}: {e}", RED)


def main():
    """Main function."""
    print_colored("PolicyCortex Linting Fix Script", CYAN)
    print_colored("===============================", CYAN)
    
    # Fix Python files
    fix_python_files()
    
    print_colored("\n✓ Linting fixes completed!", GREEN)
    print_colored("Run 'git diff' to review changes before committing.", CYAN)


if __name__ == "__main__":
    main()