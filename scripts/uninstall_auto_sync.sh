#!/bin/bash

# Uninstall Jupytext auto-sync daemon

DAEMON_PLIST="$HOME/Library/LaunchAgents/com.stockanalyze.jupytext.plist"

echo "🗑️  Uninstalling Jupytext Auto-Sync Daemon..."

# Stop and unload daemon
if launchctl list | grep -q "com.stockanalyze.jupytext"; then
    echo "🛑 Stopping daemon..."
    launchctl stop com.stockanalyze.jupytext 2>/dev/null || true
    launchctl unload "$DAEMON_PLIST" 2>/dev/null || true
fi

# Remove plist file
if [ -f "$DAEMON_PLIST" ]; then
    echo "📋 Removing daemon configuration..."
    rm "$DAEMON_PLIST"
fi

# Check if completely removed
if ! launchctl list | grep -q "com.stockanalyze.jupytext"; then
    echo "✅ Jupytext Auto-Sync Daemon uninstalled successfully!"
    echo ""
    echo "📝 **Manual sync options still available:**"
    echo "   ./scripts/sync_notebooks.sh - Sync all notebooks"
    echo "   ./scripts/watch_and_sync.sh - Manual file watcher"
else
    echo "❌ Failed to completely uninstall daemon"
    echo "   Try manually: launchctl remove com.stockanalyze.jupytext"
fi 