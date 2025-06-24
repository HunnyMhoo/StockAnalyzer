# 📚 Notebook Conversion Issues & Solutions

## 🔍 **Problem Analysis**

The Papermill .py to .ipynb conversion failures are caused by **three main issues**:

### **Issue #1: Mixed File Formats** ❌
Different .py files use incompatible formats:

- **`06_interactive_demo.py`**: Uses **Jupytext format**
  ```python
  # ---
  # jupyter:
  #   jupytext:
  #     text_representation:
  # ---
  
  # %% [markdown]
  # # Title
  
  # %%
  # Code cell
  ```

- **`01_data_collection.py`**, **`04_feature_extraction.py`**: Use **nbconvert format**
  ```python
  #!/usr/bin/env python
  # coding: utf-8
  
  # In[ ]:
  # Code cell
  
  # In[1]:
  # Another cell
  ```

### **Issue #2: Missing Dependencies** ❌
- **Papermill was NOT installed** (required for CI automation)
- **Jupytext inconsistently used** (installed but not in requirements.txt)

### **Issue #3: Tool Confusion** ❌
- Documentation mentions **Papermill** for CI
- Files set up for **Jupytext** conversion
- **These are different tools** with different workflows

---

## ✅ **Solution: Standardized Conversion Workflow**

### **Step 1: Install Missing Dependencies**
```bash
pip install papermill>=2.4.0 jupytext>=1.14.0
```

### **Step 2: Choose Consistent Format**
**Recommendation: Use Jupytext format for all .py files**

#### **Option A: Jupytext Workflow (Recommended)**
- ✅ **Bidirectional sync** (.py ↔ .ipynb)
- ✅ **Git-friendly** .py files
- ✅ **Automatic conversion**
- ✅ **Preserves metadata**

#### **Option B: Papermill-only Workflow**
- ✅ **CI automation**
- ❌ **One-way conversion** (.ipynb → execution)
- ❌ **Requires .ipynb source**

### **Step 3: Standardize Existing Files**

#### **Convert nbconvert format → Jupytext format:**
```bash
# For each .py file with nbconvert format
jupytext --set-formats ipynb,py:percent notebooks/01_data_collection.py
jupytext --set-formats ipynb,py:percent notebooks/04_feature_extraction.py
jupytext --set-formats ipynb,py:percent notebooks/05_pattern_model_training.py
```

#### **Sync .py files to .ipynb:**
```bash
jupytext --sync notebooks/*.py
```

---

## 🛠️ **Implementation Guide**

### **Method 1: Fix Current Mixed Format (Quick Fix)**

1. **Standardize to Jupytext format:**
   ```bash
   cd notebooks/
   
   # Convert nbconvert files to Jupytext format
   jupytext --to py:percent 01_data_collection.ipynb
   jupytext --to py:percent 04_feature_extraction.ipynb
   jupytext --to py:percent 05_pattern_model_training.ipynb
   ```

2. **Generate missing .ipynb files:**
   ```bash
   # For files that only have .py versions
   jupytext --to ipynb 06_interactive_demo.py
   ```

3. **Set up automatic syncing:**
   ```bash
   # Configure paired notebooks
   jupytext --set-formats ipynb,py:percent *.ipynb
   ```

### **Method 2: Papermill CI Integration**

1. **Install Papermill:**
   ```bash
   pip install papermill>=2.4.0
   ```

2. **Use .ipynb files as source:**
   ```bash
   # Execute notebook with parameters
   papermill notebooks/06_interactive_demo.ipynb output.ipynb \
     -p SELECTED_TICKERS '["0700.HK", "0005.HK"]' \
     -p QUICK_MODE True
   ```

3. **CI Pipeline example:**
   ```yaml
   # .github/workflows/notebook-tests.yml
   - name: Test Interactive Demo
     run: |
       papermill notebooks/06_interactive_demo.ipynb test_output.ipynb \
         -p QUICK_MODE True
   ```

---

## 🔧 **Recommended Workflow**

### **For Development:**
1. **Edit .ipynb files** in Jupyter
2. **Jupytext auto-syncs** to .py files
3. **Commit both formats** to git
4. **nbstripout removes** output from .ipynb

### **For CI:**
1. **Use Papermill** to execute .ipynb files
2. **Pass parameters** for different test scenarios
3. **Check outputs** for validation

### **Configuration Files:**

#### **jupytext.toml** (project root):
```toml
[tool.jupytext]
formats = "ipynb,py:percent"
```

#### **.gitattributes** (already exists):
```
*.ipynb filter=nbstripout
```

---

## 📊 **File Status & Actions Needed**

| File | Current Format | Action Required | Status |
|------|----------------|-----------------|--------|
| `06_interactive_demo.py` | ✅ Jupytext | None | Ready |
| `01_data_collection.py` | ❌ nbconvert | Convert to Jupytext | Needs fix |
| `04_feature_extraction.py` | ❌ nbconvert | Convert to Jupytext | Needs fix |
| `05_pattern_model_training.py` | ❌ nbconvert | Convert to Jupytext | Needs fix |
| `06_pattern_scanning.py` | ❌ nbconvert | Convert to Jupytext | Needs fix |

---

## 🚀 **Quick Fix Commands**

Run these commands to fix all conversion issues:

```bash
# 1. Install dependencies
pip install papermill>=2.4.0

# 2. Navigate to notebooks directory
cd notebooks/

# 3. Convert all .py files to Jupytext format
for file in 01_data_collection 04_feature_extraction 05_pattern_model_training 06_pattern_scanning; do
    if [ -f "${file}.ipynb" ]; then
        echo "Converting ${file}.ipynb to Jupytext format..."
        jupytext --to py:percent "${file}.ipynb"
    fi
done

# 4. Generate .ipynb files for Jupytext-only .py files
jupytext --to ipynb 06_interactive_demo.py

# 5. Set up automatic syncing for all notebooks
jupytext --set-formats ipynb,py:percent *.ipynb

# 6. Verify conversion
echo "✅ Conversion complete. Files ready for Papermill."
```

---

## 🧪 **Testing the Fix**

After running the fix commands, test Papermill conversion:

```bash
# Test basic conversion
papermill 06_interactive_demo.ipynb test_output.ipynb -p QUICK_MODE True

# Test with parameters
papermill 01_data_collection.ipynb test_data_collection.ipynb

# Verify all notebooks can be executed
for notebook in *.ipynb; do
    echo "Testing $notebook..."
    papermill "$notebook" "test_$(basename "$notebook")" --log-output
done
```

---

## 📚 **Additional Resources**

- **Jupytext Documentation**: https://jupytext.readthedocs.io/
- **Papermill Documentation**: https://papermill.readthedocs.io/
- **nbstripout Setup**: https://github.com/kynan/nbstripout

---

**🎯 Result**: After implementing these fixes, all notebooks will have consistent formats and Papermill will work reliably for CI automation. 