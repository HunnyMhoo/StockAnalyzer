# 🔄 Jupytext Auto-Sync Guide

Complete guide for automatically converting `.py` files to `.ipynb` files in the StockAnalyze project.

## 📋 Quick Reference

| Method | When to Use | Automation Level |
|--------|-------------|------------------|
| [Manual Sync](#-method-1-manual-sync) | One-time conversion | Manual |
| [Git Hooks](#-method-2-git-hooks-automatic) | Auto-sync on commit | Automatic |
| [VS Code Tasks](#-method-3-vs-codecursor-integration) | IDE integration | Semi-automatic |
| [Batch Script](#-method-4-batch-script) | Sync all files at once | Manual |
| [Watch Script](#-method-5-real-time-watch-script) | Real-time monitoring | Automatic |

---

## 🔧 Method 1: Manual Sync

**Use when:** You need to sync specific files on demand.

```bash
# Sync a single file
jupytext --sync notebooks/06_interactive_demo.py

# Sync all .py files in notebooks directory
jupytext --sync notebooks/*.py

# Sync and convert to specific format
jupytext --to ipynb notebooks/06_interactive_demo.py
```

**Pros:** Simple, precise control  
**Cons:** Manual process, easy to forget

---

## ⚡ Method 2: Git Hooks (Automatic)

**Use when:** You want automatic sync every time you commit code.

The pre-commit hook is already installed at `.git/hooks/pre-commit`.

### How it works:
1. Every time you run `git commit`
2. Hook automatically syncs all `.py` files with Jupytext headers
3. Stages the updated `.ipynb` files
4. Completes the commit

### Test the hook:
```bash
# Make a small change to any .py file
echo "# Test comment" >> notebooks/06_interactive_demo.py

# Commit - watch the auto-sync happen
git add notebooks/06_interactive_demo.py
git commit -m "Test auto-sync"
```

**Pros:** Fully automatic, ensures consistency  
**Cons:** Adds time to commits, may sync unwanted files

---

## 🎨 Method 3: VS Code/Cursor Integration

**Use when:** You want IDE integration with keyboard shortcuts.

Tasks are configured in `.vscode/tasks.json`.

### Available tasks:
- **Sync Current File:** `Cmd+Shift+P` → "Tasks: Run Task" → "Jupytext: Sync Current File"
- **Sync All Notebooks:** `Cmd+Shift+P` → "Tasks: Run Task" → "Jupytext: Sync All Notebooks"
- **Auto-sync on Save:** Automatically syncs when you save a file

### Set up keyboard shortcuts:
Add to your VS Code `keybindings.json`:
```json
[
    {
        "key": "cmd+shift+j",
        "command": "workbench.action.tasks.runTask",
        "args": "Jupytext: Sync Current File"
    }
]
```

**Pros:** IDE integration, keyboard shortcuts  
**Cons:** Editor-specific, requires manual trigger

---

## 🚀 Method 4: Batch Script

**Use when:** You want to sync all files at once with detailed feedback.

```bash
# Run the batch sync script
./scripts/sync_notebooks.sh
```

### What it does:
- ✅ Syncs all `.py` files with Jupytext headers
- ⏭️ Skips files without headers (like `common_setup.py`)
- 📊 Shows git status of changed files
- 📈 Provides processing statistics

**Pros:** Comprehensive, informative output  
**Cons:** Manual execution required

---

## 👀 Method 5: Real-time Watch Script

**Use when:** You want automatic sync as you work.

```bash
# Start the file watcher (runs in background)
./scripts/watch_and_sync.sh

# Stop with Ctrl+C
```

### Features:
- 🔍 Monitors all `.py` files in `notebooks/` directory
- 🔄 Auto-syncs when files change
- ⚡ Uses `fswatch` on macOS for efficiency
- 🔄 Falls back to polling if `fswatch` not available

### Install fswatch for better performance:
```bash
brew install fswatch
```

**Pros:** Real-time sync, works while you code  
**Cons:** Uses system resources, may be too aggressive

---

## 🎯 Recommended Workflow

### For Daily Development:
1. **Start with git hooks** for automatic commit-time sync
2. **Use batch script** when you want to sync everything manually
3. **Use manual sync** for specific files when needed

### For Intensive Notebook Work:
1. **Start watch script** at beginning of session
2. **Work normally** - files sync automatically
3. **Stop watch script** when done

### For Team Collaboration:
1. **Ensure git hooks** are installed for all team members
2. **Use batch script** before major commits
3. **Document sync status** in commit messages

---

## 🔍 Troubleshooting

### "Command not found: jupytext"
```bash
pip install jupytext
```

### Files not syncing automatically
Check if file has Jupytext header:
```bash
head -20 notebooks/your_file.py | grep jupytext
```

### Git hook not working
```bash
# Check if hook is executable
ls -la .git/hooks/pre-commit

# Make executable if needed
chmod +x .git/hooks/pre-commit
```

### Watch script issues on macOS
```bash
# Install fswatch for better performance
brew install fswatch
```

---

## 📁 File Structure

```
StockAnalyze/
├── .git/hooks/
│   └── pre-commit              # Auto-sync on git commit
├── .vscode/
│   └── tasks.json              # VS Code integration
├── scripts/
│   ├── sync_notebooks.sh       # Batch sync script
│   └── watch_and_sync.sh       # Real-time watcher
├── jupytext.toml               # Jupytext configuration
└── notebooks/
    ├── *.py                    # Source files (edit these)
    └── *.ipynb                 # Generated files (auto-synced)
```

---

## ⚙️ Configuration

Your `jupytext.toml` is configured for:
- **Format pairing:** `.py` ↔ `.ipynb`
- **Cell format:** Percent format (`# %%`)
- **Metadata filtering:** Clean notebooks without clutter

---

## 🎉 Success!

You now have **5 different methods** to automatically convert `.py` files to `.ipynb`:

1. ✅ **Git hooks** installed and working
2. ✅ **VS Code tasks** configured
3. ✅ **Batch script** ready to use
4. ✅ **Watch script** for real-time sync
5. ✅ **Manual commands** for precise control

Choose the method that best fits your workflow!

*Context added by Giga data-flow-pipeline* 