# 🔄 Jupytext Auto-Sync Guide

This guide shows you **4 methods** to automatically sync `.py` files to `.ipynb` notebooks when files are updated.

## 🎯 Method 1: File Watcher (Recommended)

**Best for**: Active development sessions

### Setup (One-time):
```bash
# Install fswatch (macOS)
brew install fswatch

# Make script executable
chmod +x scripts/watch_and_sync.sh
```

### Usage:
```bash
# Start the watcher (runs in background)
./scripts/watch_and_sync.sh

# In another terminal, edit any .py file in notebooks/
# The watcher will automatically sync to .ipynb

# Stop the watcher
Ctrl+C  # or pkill -f fswatch
```

**✅ Pros**: Real-time sync, efficient, works across all notebooks  
**❌ Cons**: Requires running process, macOS only (fswatch)

---

## 🔧 Method 2: Git Hooks (Set and Forget)

**Best for**: Automatic sync on commits

### Setup:
```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "🔄 Auto-syncing notebooks before commit..."
./scripts/sync_notebooks.sh
git add notebooks/*.ipynb
EOF

chmod +x .git/hooks/pre-commit
```

**✅ Pros**: Automatic on every commit, no manual intervention  
**❌ Cons**: Only syncs on commit, not during development

---

## 📝 Method 3: Editor Integration

**Best for**: IDE-based development

### VS Code Setup:
1. Install "Jupytext for Notebooks" extension
2. Add to `settings.json`:
```json
{
    "jupytext.formats": "ipynb,py:percent",
    "files.watcherExclude": {
        "**/notebooks/*.ipynb": false
    }
}
```

### PyCharm Setup:
1. Install "Jupyter" plugin
2. Configure File Watchers:
   - File type: `*.py`
   - Scope: `notebooks/`
   - Program: `jupytext`
   - Arguments: `--sync $FilePath$`

**✅ Pros**: Integrated with editor, automatic on save  
**❌ Cons**: Editor-specific setup required

---

## ⚡ Method 4: Manual Batch Sync

**Best for**: Occasional sync or CI/CD

### Usage:
```bash
# Sync all notebooks at once
./scripts/sync_notebooks.sh

# Sync specific file
jupytext --sync notebooks/06_interactive_demo.py

# Sync all files in directory
jupytext --sync notebooks/*.py
```

**✅ Pros**: Simple, reliable, works everywhere  
**❌ Cons**: Manual trigger required

---

## 🏗️ Current Project Configuration

Your project is already configured with:

### `jupytext.toml`:
```toml
# Automatically sync .py and .ipynb files
formats = "ipynb,py:percent"

# Default configuration
notebook_metadata_filter = "all,-language_info,-toc,-latex_envs"
cell_metadata_filter = "-all"
```

### Available Scripts:
- `scripts/watch_and_sync.sh` - Real-time file watcher
- `scripts/sync_notebooks.sh` - Batch sync all notebooks

---

## 🚀 Quick Start (Recommended Workflow)

1. **For active development**:
   ```bash
   # Terminal 1: Start file watcher
   ./scripts/watch_and_sync.sh
   
   # Terminal 2: Edit .py files
   code notebooks/06_interactive_demo.py
   # Files automatically sync to .ipynb
   ```

2. **For occasional work**:
   ```bash
   # Edit files, then sync manually
   ./scripts/sync_notebooks.sh
   ```

3. **For team collaboration**:
   ```bash
   # Set up git hook (one-time)
   cat > .git/hooks/pre-commit << 'EOF'
   #!/bin/bash
   ./scripts/sync_notebooks.sh
   git add notebooks/*.ipynb
   EOF
   chmod +x .git/hooks/pre-commit
   ```

---

## 🔍 Verification

Check if files are synced:
```bash
# Compare timestamps
ls -la notebooks/06_interactive_demo.*

# Should show same or very close timestamps:
# -rw-r--r-- 1 user staff 15481 Jun 25 20:40 06_interactive_demo.ipynb
# -rw-r--r-- 1 user staff 11155 Jun 25 20:40 06_interactive_demo.py
```

---

## 🛠️ Troubleshooting

### File not syncing?
```bash
# Check if file has Jupytext header
head -20 notebooks/your_file.py | grep jupytext

# Should show:
# jupyter:
#   jupytext:
#     text_representation:
```

### Watcher not working?
```bash
# Check if fswatch is installed
which fswatch

# Check if watcher is running
ps aux | grep fswatch

# Restart watcher
pkill -f fswatch
./scripts/watch_and_sync.sh
```

### Manual sync fails?
```bash
# Check Jupytext installation
jupytext --version

# Try direct sync
jupytext --sync notebooks/your_file.py --verbose
```

---

## 📊 Summary

| Method | Auto-Sync | Setup | Best For |
|--------|-----------|-------|----------|
| **File Watcher** | ✅ Real-time | Easy | Active development |
| **Git Hooks** | ✅ On commit | Medium | Team collaboration |
| **Editor Integration** | ✅ On save | Complex | IDE users |
| **Manual Batch** | ❌ Manual | None | Occasional use |

**Recommendation**: Use **File Watcher** for development + **Git Hooks** for commits.

*Context added by Giga data-flow-pipeline* 