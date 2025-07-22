#!/bin/bash

# Install automatic Jupytext sync daemon for macOS
# This ensures .py and .ipynb files stay synchronized automatically

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_FILE="$PROJECT_ROOT/scripts/com.stockanalyze.jupytext.plist"
DAEMON_PLIST="$HOME/Library/LaunchAgents/com.stockanalyze.jupytext.plist"

echo "🚀 Installing Jupytext Auto-Sync Daemon..."
echo "   Project: $PROJECT_ROOT"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Check if fswatch is installed
if ! command -v fswatch >/dev/null 2>&1; then
    echo "⚠️  fswatch is not installed. Installing via Homebrew..."
    if command -v brew >/dev/null 2>&1; then
        brew install fswatch
    else
        echo "❌ Homebrew not found. Please install fswatch manually:"
        echo "   brew install fswatch"
        exit 1
    fi
fi

# Check if jupytext is available
if ! command -v jupytext >/dev/null 2>&1; then
    echo "❌ jupytext is not available in PATH. Please ensure it's installed:"
    echo "   pip install jupytext"
    exit 1
fi

# Stop existing daemon if running
if launchctl list | grep -q "com.stockanalyze.jupytext"; then
    echo "🛑 Stopping existing daemon..."
    launchctl unload "$DAEMON_PLIST" 2>/dev/null || true
fi

# Copy plist file to LaunchAgents
echo "📋 Installing daemon configuration..."
cp "$PLIST_FILE" "$DAEMON_PLIST"

# Load and start the daemon
echo "🔄 Starting daemon..."
launchctl load "$DAEMON_PLIST"
launchctl start com.stockanalyze.jupytext

# Wait a moment and check status
sleep 2

if launchctl list | grep -q "com.stockanalyze.jupytext"; then
    echo "✅ Jupytext Auto-Sync Daemon installed and running!"
    echo ""
    echo "🎯 **What this does:**"
    echo "   - Automatically syncs .py ↔ .ipynb files when you edit them"
    echo "   - Runs in the background (survives restarts)"
    echo "   - Logs activity to: $PROJECT_ROOT/logs/"
    echo ""
    echo "🔧 **Management commands:**"
    echo "   Start:   launchctl start com.stockanalyze.jupytext"
    echo "   Stop:    launchctl stop com.stockanalyze.jupytext"
    echo "   Status:  launchctl list | grep jupytext"
    echo "   Logs:    tail -f $PROJECT_ROOT/logs/jupytext_sync.log"
    echo ""
    echo "🗑️  **To uninstall:**"
    echo "   $PROJECT_ROOT/scripts/uninstall_auto_sync.sh"
else
    echo "❌ Failed to start daemon. Check logs:"
    echo "   $PROJECT_ROOT/logs/jupytext_sync_error.log"
    exit 1
fi 