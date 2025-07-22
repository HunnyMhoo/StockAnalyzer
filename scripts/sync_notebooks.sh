#!/bin/bash

# Script to sync all Jupyter notebooks with their Python counterparts
# This ensures .py and .ipynb files stay consistent

echo "🔄 Syncing all notebooks..."

# Sync core workflow notebooks
echo "📁 Syncing core workflow notebooks..."
jupytext --sync notebooks/core_workflow/*.ipynb

# Sync example notebooks  
echo "📁 Syncing example notebooks..."
jupytext --sync notebooks/examples/*.ipynb

# Sync integration test notebooks
echo "📁 Syncing integration test notebooks..."
jupytext --sync tests/integration/*.ipynb

echo "✅ All notebooks synchronized successfully!"
echo "📝 Note: If you made changes to .py files, the .ipynb files are now updated."
echo "📝 Note: If you made changes to .ipynb files, the .py files are now updated." 