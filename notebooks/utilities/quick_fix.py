"""
Quick fix utility for import issues in notebooks.
Run this cell first in any notebook that has import problems.
"""

import sys
import os
from pathlib import Path

def fix_imports():
    """Fix import paths for notebooks that can't find utilities module."""
    # Get current working directory
    current_dir = Path.cwd()
    
    # Find notebooks directory
    if current_dir.name == 'notebooks':
        notebooks_dir = current_dir
        project_root = current_dir.parent
    elif current_dir.parent.name == 'notebooks':
        notebooks_dir = current_dir.parent
        project_root = current_dir.parent.parent
    else:
        # Search for notebooks directory in parent paths
        notebooks_dir = None
        project_root = current_dir
        for parent in current_dir.parents:
            if (parent / 'notebooks').exists():
                notebooks_dir = parent / 'notebooks'
                project_root = parent
                break
        
        if notebooks_dir is None:
            notebooks_dir = current_dir / 'notebooks'
            project_root = current_dir
    
    # Add necessary paths
    paths_to_add = [
        str(notebooks_dir),
        str(project_root / 'src'),
        str(project_root / 'stock_analyzer'),
        str(project_root)
    ]
    
    for path in paths_to_add:
        if path not in sys.path and os.path.exists(path):
            sys.path.insert(0, path)
    
    print(f"✅ Fixed import paths!")
    print(f"📂 Notebooks directory: {notebooks_dir}")
    print(f"📂 Project root: {project_root}")
    
    # Test the import
    try:
        from utilities.common_setup import setup_notebook
        print("✅ utilities.common_setup can now be imported!")
        return True
    except ImportError as e:
        print(f"❌ Still having import issues: {e}")
        return False

if __name__ == "__main__":
    fix_imports() 