#!/bin/bash
# Watch .py files and automatically sync to .ipynb when they change

echo "🔍 Starting file watcher for automatic Jupytext sync..."
echo "   Watching: notebooks/*.py"
echo "   Press Ctrl+C to stop"

# Change to project root
cd "$(dirname "$0")/.."

# Check if fswatch is available (macOS)
if command -v fswatch >/dev/null 2>&1; then
    echo "   Using fswatch for file monitoring"
    
    fswatch -o notebooks/*.py | while read f; do
        echo "🔄 Change detected, syncing notebooks..."
        
        for py_file in notebooks/*.py; do
            if [ -f "$py_file" ] && head -20 "$py_file" | grep -q "jupytext:"; then
                echo "   📝 Syncing: $(basename "$py_file")"
                jupytext --sync "$py_file" 2>/dev/null
            fi
        done
        
        echo "✅ Sync complete at $(date '+%H:%M:%S')"
    done
    
# Fallback to polling method
else
    echo "   Using polling method (install fswatch for better performance)"
    echo "   Install with: brew install fswatch"
    
    # Store initial checksums
    declare -A checksums
    
    while true; do
        changed=false
        
        for py_file in notebooks/*.py; do
            if [ -f "$py_file" ] && head -20 "$py_file" | grep -q "jupytext:"; then
                current_checksum=$(md5 -q "$py_file" 2>/dev/null || md5sum "$py_file" 2>/dev/null | cut -d' ' -f1)
                
                if [ "${checksums[$py_file]}" != "$current_checksum" ]; then
                    echo "🔄 Change detected in $(basename "$py_file")"
                    jupytext --sync "$py_file" 2>/dev/null
                    checksums[$py_file]="$current_checksum"
                    changed=true
                fi
            fi
        done
        
        if [ "$changed" = true ]; then
            echo "✅ Sync complete at $(date '+%H:%M:%S')"
        fi
        
        sleep 2
    done
fi 