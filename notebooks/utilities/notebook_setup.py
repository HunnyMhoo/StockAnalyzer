"""
Simple notebook setup utility for path configuration

This module handles the Python path setup needed for notebooks to import
utilities and other modules correctly, regardless of their location.
"""

import sys
import os
from pathlib import Path

def setup_notebook_paths():
    """
    Setup Python paths for notebook imports.
    
    This function ensures that notebooks can import:
    - utilities.common_setup
    - src modules
    - stock_analyzer modules
    
    Call this at the beginning of any notebook before importing other modules.
    """
    current_dir = Path.cwd()
    
    # Find the notebooks directory
    if current_dir.name == 'notebooks':
        notebooks_dir = current_dir
        project_root = current_dir.parent
    elif current_dir.parent.name == 'notebooks':
        notebooks_dir = current_dir.parent
        project_root = current_dir.parent.parent
    else:
        # Look for notebooks directory in parent paths
        notebooks_dir = None
        project_root = current_dir
        for parent in current_dir.parents:
            if (parent / 'notebooks').exists():
                notebooks_dir = parent / 'notebooks'
                project_root = parent
                break
        
        if notebooks_dir is None:
            # Fallback: assume we're in project root
            notebooks_dir = current_dir / 'notebooks'
            project_root = current_dir
    
    # Add paths to sys.path
    paths_to_add = [
        str(notebooks_dir),  # For utilities.common_setup
        str(project_root / 'src'),  # For src modules
        str(project_root / 'stock_analyzer'),  # For stock_analyzer modules
        str(project_root)  # For project root imports
    ]
    
    for path in paths_to_add:
        if path not in sys.path and os.path.exists(path):
            sys.path.insert(0, path)
    
    return {
        'notebooks_dir': notebooks_dir,
        'project_root': project_root,
        'paths_added': paths_to_add
    }

def quick_setup():
    """
    Quick setup function that configures paths and imports common_setup.
    
    Returns the setup_notebook function from common_setup for convenience.
    """
    setup_notebook_paths()
    
    try:
        from utilities.common_setup import setup_notebook
        return setup_notebook
    except ImportError as e:
        print(f"❌ Could not import common_setup: {e}")
        print("🔍 Available paths:")
        for path in sys.path[:5]:  # Show first 5 paths
            print(f"   {path}")
        raise

# For backwards compatibility and convenience
def setup_paths():
    """Alias for setup_notebook_paths()"""
    return setup_notebook_paths() 