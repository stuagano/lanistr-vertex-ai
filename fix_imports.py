#!/usr/bin/env python3
"""
Fix relative imports in LANISTR project
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix relative imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix from utils. imports
    content = re.sub(r'from utils\.', 'from lanistr.utils.', content)
    
    # Fix from dataset. imports
    content = re.sub(r'from dataset\.', 'from lanistr.dataset.', content)
    
    # Fix from model. imports
    content = re.sub(r'from model\.', 'from lanistr.model.', content)
    
    # Fix from third_party. imports
    content = re.sub(r'from third_party\.', 'from lanistr.third_party.', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed imports in {file_path}")

def main():
    """Fix imports in all Python files."""
    lanistr_dir = Path("lanistr")
    
    # Find all Python files
    python_files = list(lanistr_dir.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files to fix")
    
    for file_path in python_files:
        try:
            fix_imports_in_file(file_path)
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")

if __name__ == "__main__":
    main() 