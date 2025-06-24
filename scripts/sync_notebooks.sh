#!/bin/bash
# Batch sync all .py files to .ipynb in the notebooks directory

echo "🔄 Syncing all notebook files..."

# Change to project root
cd "$(dirname "$0")/.."

# Counter for processed files
count=0

# Process all .py files in notebooks directory
for py_file in notebooks/*.py; do
    if [ -f "$py_file" ]; then
        # Check if file has Jupytext header
        if head -20 "$py_file" | grep -q "jupytext:"; then
            echo "   📝 Syncing: $(basename "$py_file")"
            jupytext --sync "$py_file"
            ((count++))
        else
            echo "   ⏭️  Skipping: $(basename "$py_file") (no Jupytext header)"
        fi
    fi
done

echo "✅ Processed $count notebook files"

# Optional: Show git status if in a git repository
if [ -d ".git" ]; then
    echo ""
    echo "📊 Git Status:"
    git status --porcelain notebooks/*.ipynb | head -10
fi 