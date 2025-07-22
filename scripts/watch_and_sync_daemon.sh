#!/bin/bash

# Daemon-optimized Jupytext sync watcher
# This version is designed to run as a background service

# Enable error handling
set -e

# Set up logging
LOG_DIR="$(dirname "$0")/../logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/jupytext_sync.log"
ERROR_LOG="$LOG_DIR/jupytext_sync_error.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$ERROR_LOG"
}

# Change to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

log "🚀 Starting Jupytext Auto-Sync Daemon"
log "   Project: $PROJECT_ROOT"
log "   Log file: $LOG_FILE"
log "   Error log: $ERROR_LOG"

# Check dependencies
if ! command -v fswatch >/dev/null 2>&1; then
    log_error "fswatch not found. Install with: brew install fswatch"
    exit 1
fi

if ! command -v jupytext >/dev/null 2>&1; then
    log_error "jupytext not found. Install with: pip install jupytext"
    exit 1
fi

# Function to sync a file
sync_file() {
    local file="$1"
    local basename_file=$(basename "$file")
    
    # Check if file has Jupytext metadata
    if head -20 "$file" | grep -q "jupytext:"; then
        log "   📝 Syncing: $basename_file"
        
        if jupytext --sync "$file" 2>>"$ERROR_LOG"; then
            log "   ✅ Synced: $basename_file"
        else
            log_error "Failed to sync: $basename_file"
        fi
    else
        log "   ⏭️  Skipping: $basename_file (no jupytext metadata)"
    fi
}

# Function to sync all notebooks
sync_all_notebooks() {
    log "🔄 Syncing all notebooks..."
    
    local sync_count=0
    local error_count=0
    
    # Find all .py files with Jupytext metadata
    while IFS= read -r -d '' py_file; do
        if [[ -f "$py_file" ]]; then
            sync_file "$py_file"
            ((sync_count++))
        fi
    done < <(find notebooks -name "*.py" -print0)
    
    log "✅ Sync complete: $sync_count files processed"
}

# Cleanup function
cleanup() {
    log "🛑 Shutting down Jupytext Auto-Sync Daemon"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Initial sync
sync_all_notebooks

# Start file watcher
log "🔍 Starting file watcher..."
log "   Watching: notebooks/**/*.py"

# Use fswatch to monitor file changes
fswatch -r -e ".*" -i "\\.py$" --event Created --event Updated --event Renamed notebooks/ | while read -r changed_file; do
    log "🔄 Change detected: $(basename "$changed_file")"
    
    # Small delay to avoid rapid-fire syncing
    sleep 1
    
    # Sync the specific file
    if [[ -f "$changed_file" ]]; then
        sync_file "$changed_file"
    fi
    
    # Also sync any related .ipynb file
    ipynb_file="${changed_file%.py}.ipynb"
    if [[ -f "$ipynb_file" ]]; then
        log "   🔄 Also syncing related .ipynb file"
        sync_file "$ipynb_file"
    fi
    
done

# This should never be reached, but just in case
log "🛑 File watcher stopped unexpectedly" 