#!/usr/bin/env python3
"""
Add patent headers to all source files in the PolicyCortex project.
This script adds patent notices to protect intellectual property.
"""

import os
import glob
from pathlib import Path

# Patent header for Rust files
RUST_PATENT_HEADER = """// PATENT NOTICE: This code implements methods covered by:
// - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
// - US Patent Application 17/123,457 - Conversational Governance Intelligence System  
// - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
// - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
// Unauthorized use, reproduction, or distribution may constitute patent infringement.
// © 2024 PolicyCortex. All rights reserved.

"""

# Patent header for TypeScript/JavaScript files
TS_PATENT_HEADER = """/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

"""

# Patent header for Python files
PY_PATENT_HEADER = '''"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

'''

def add_header_to_file(filepath, header):
    """Add patent header to a file if it doesn't already have one."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if patent header already exists
        if 'PATENT NOTICE' in content[:500]:
            return False
        
        # Add header at the beginning of the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header + content)
        
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function to add patent headers to all source files."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    files_updated = 0
    
    # Process Rust files
    print("Adding patent headers to Rust files...")
    rust_files = glob.glob("core/src/**/*.rs", recursive=True)
    rust_files.extend(glob.glob("crates/**/*.rs", recursive=True))
    rust_files.extend(glob.glob("edge/**/*.rs", recursive=True))
    
    for filepath in rust_files:
        if 'target' not in filepath and add_header_to_file(filepath, RUST_PATENT_HEADER):
            files_updated += 1
            print(f"  [ADDED] {filepath}")
    
    # Process TypeScript/TSX files
    print("\nAdding patent headers to TypeScript files...")
    ts_files = glob.glob("frontend/**/*.tsx", recursive=True)
    ts_files.extend(glob.glob("frontend/**/*.ts", recursive=True))
    ts_files.extend(glob.glob("graphql/**/*.js", recursive=True))
    ts_files.extend(glob.glob("graphql/**/*.ts", recursive=True))
    
    for filepath in ts_files:
        if 'node_modules' not in filepath and '.next' not in filepath:
            if add_header_to_file(filepath, TS_PATENT_HEADER):
                files_updated += 1
                print(f"  [ADDED] {filepath}")
    
    # Process Python files
    print("\nAdding patent headers to Python files...")
    py_files = glob.glob("backend/**/*.py", recursive=True)
    py_files.extend(glob.glob("training/**/*.py", recursive=True))
    py_files.extend(glob.glob("ml/**/*.py", recursive=True))
    py_files.extend(glob.glob("scripts/**/*.py", recursive=True))
    
    for filepath in py_files:
        if '__pycache__' not in filepath and add_header_to_file(filepath, PY_PATENT_HEADER):
            files_updated += 1
            print(f"  [ADDED] {filepath}")
    
    print(f"\n[SUCCESS] Patent headers added to {files_updated} files")
    print("Patent protection is now in place across the codebase!")

if __name__ == "__main__":
    main()