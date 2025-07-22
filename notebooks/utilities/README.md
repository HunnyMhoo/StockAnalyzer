# Notebook Utilities

This directory contains utilities for notebook setup and configuration.

## Quick Start

For new notebooks, use the simplified setup:

```python
# Simple approach - add this at the beginning of any notebook
import sys
from pathlib import Path

# Add notebooks directory to path
notebooks_dir = Path.cwd()
if notebooks_dir.name != 'notebooks':
    notebooks_dir = notebooks_dir.parent if notebooks_dir.parent.name == 'notebooks' else notebooks_dir.parent.parent / 'notebooks'
    
sys.path.insert(0, str(notebooks_dir))

# Now you can import utilities
from utilities.common_setup import setup_notebook
```

## Alternative: Using notebook_setup.py

For even simpler setup, you can use the provided helper:

```python
# Even simpler approach
import sys
from pathlib import Path

# Add notebooks directory to path
notebooks_dir = Path.cwd()
if notebooks_dir.name != 'notebooks':
    notebooks_dir = notebooks_dir.parent if notebooks_dir.parent.name == 'notebooks' else notebooks_dir.parent.parent / 'notebooks'
sys.path.insert(0, str(notebooks_dir))

# Use the helper
from utilities.notebook_setup import quick_setup
setup_notebook = quick_setup()

# Now run standard setup
validation = setup_notebook()
```

## Files

- `common_setup.py` - Main setup utilities with path configuration, warnings, display options, etc.
- `notebook_setup.py` - Simple path configuration helper for notebooks
- `README.md` - This documentation

## Common Issues

**ModuleNotFoundError: No module named 'utilities'**

This happens when the notebook can't find the utilities module. Add the path setup code shown above at the beginning of your notebook.

**Why does this happen?**

When notebooks run from subdirectories like `notebooks/core_workflow/`, Python doesn't automatically know where to find the `utilities` module in `notebooks/utilities/`. The path setup code adds the notebooks directory to Python's module search path.

## Project Structure

```
notebooks/
├── utilities/           # This directory
│   ├── common_setup.py  # Main utilities
│   ├── notebook_setup.py # Path helper
│   └── README.md        # This file
├── core_workflow/       # Main workflow notebooks
├── examples/           # Example notebooks
└── ...
``` 