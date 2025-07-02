# 📚 Hong Kong Stock Analysis - Notebook Organization

This directory contains the complete notebook ecosystem for Hong Kong stock pattern analysis, now organized into a clean, maintainable structure.

## 🎯 **Quick Start**

**New Users (15 minutes):**
```bash
# Start with the quick demo
jupyter notebook examples/quick_start_demo.ipynb
```

**Beginner Users (25-100 stocks):**
```bash
# Follow the progressive workflow
jupyter notebook core_workflow/02_data_collection_starter.ipynb
```

**Professional Users (100-1000+ stocks):**
```bash
# Use enterprise-grade features
jupyter notebook core_workflow/02_data_collection_enterprise.ipynb
```

## 📁 **Directory Structure**

### 🎯 **core_workflow/** - Main Production Workflow
*Sequential workflow for complete stock analysis*

| Notebook | Purpose | Time | Stocks |
|----------|---------|------|--------|
| `00_workflow_index.py` | Overview and navigation | 5 min | - |
| `01_data_collection.py` | Basic data collection | 10 min | 10-20 |
| `02_data_collection_starter.py` | **Beginner-friendly collection** | 15-35 min | 10-100 |
| `02_data_collection_enterprise.py` | **Enterprise-grade collection** | 15-180 min | 100-1000+ |
| `03_feature_extraction.py` | Technical analysis & features | 10-20 min | Any |
| `04_pattern_training.py` | ML model training | 15-30 min | Any |
| `05_pattern_detection.py` | Automated pattern scanning | 5-15 min | Any |
| `06_interactive_analysis.py` | Interactive analysis UI | 10-20 min | Any |
| `07_visualization.py` | Advanced charts & reports | 10-15 min | Any |
| `08_signal_analysis.py` | Performance tracking | 10-15 min | Any |

### 🎮 **examples/** - Learning and Tutorials
*Quick demos and educational content*

| Notebook | Purpose | Time | Difficulty |
|----------|---------|------|------------|
| `quick_start_demo.py` | **15-minute complete demo** | 15 min | Beginner |
| `pattern_labeling_tutorial.py` | Pattern labeling guide | 20 min | Intermediate |

### 🔧 **utilities/** - Shared Components
*Reusable utilities and configuration*

| File | Purpose |
|------|---------|
| `common_setup.py` | Shared setup and configuration |

### 📦 **archived/** - Legacy Files
*Previous versions kept for reference*

- Old data collection notebooks (consolidated)
- Legacy implementations
- Development history

## 🚀 **Recommended Learning Paths**

### 🔰 **Beginner Path**
1. **Start Here**: `examples/quick_start_demo.py` (15 min)
2. **Learn Collection**: `core_workflow/02_data_collection_starter.py` (25-50 stocks)
3. **Explore Features**: `core_workflow/03_feature_extraction.py`
4. **Try Interactive**: `core_workflow/06_interactive_analysis.py`

### 🏢 **Professional Path**
1. **Quick Overview**: `examples/quick_start_demo.py` (15 min)
2. **Scale Up**: `core_workflow/02_data_collection_enterprise.py` (100-1000+ stocks)
3. **Train Models**: `core_workflow/04_pattern_training.py`
4. **Automate Detection**: `core_workflow/05_pattern_detection.py`
5. **Track Performance**: `core_workflow/08_signal_analysis.py`

### 🎓 **Learning Path**
1. **Demo**: `examples/quick_start_demo.py`
2. **Tutorial**: `examples/pattern_labeling_tutorial.py`
3. **Progressive**: Work through `core_workflow/` 01-08 in order

## 📊 **What Changed** (Migration Guide)

### ✅ **Consolidated Notebooks**
- `02_basic_data_collection.py` + `02_advanced_data_collection.py` → `02_data_collection_starter.py`
- `02_bulk_data_collection*.py` → Archived (redundant)
- `06_interactive_demo.py` → `06_interactive_analysis.py` (renamed)
- `07_pattern_match_visualization.py` → `07_visualization.py` (simplified)

### 🗂️ **Reorganized Structure**
- **Test files** moved to `tests/integration/`
- **Examples** moved to `examples/`
- **Utilities** moved to `utilities/`
- **Legacy files** moved to `archived/`

### 🔄 **Updated Import Paths**
```python
# OLD
from notebooks.common_setup import setup_notebook

# NEW
from notebooks.utilities.common_setup import setup_notebook
```

## 🛠️ **Technical Details**

### **Size Limits**
- **Core workflows**: ≤500 lines (maintainable)
- **Examples**: ≤300 lines (focused)
- **Utilities**: ≤200 lines (modular)

### **Performance Standards**
- **Starter**: 10-100 stocks, 15-35 minutes
- **Enterprise**: 100-1000+ stocks, 15-180 minutes
- **Examples**: 5-15 stocks, 10-20 minutes

### **Code Quality**
- ✅ Jupytext format for version control
- ✅ Consistent naming conventions
- ✅ Progressive complexity
- ✅ Comprehensive error handling
- ✅ Performance monitoring

## 🧪 **Testing**

### **Integration Tests**
```bash
# Run integration tests
pytest tests/integration/
```

### **Manual Verification**
```bash
# Verify notebooks work
jupyter notebook examples/quick_start_demo.ipynb  # Should complete in 15 min
jupyter notebook core_workflow/02_data_collection_starter.ipynb  # Should handle 25 stocks
```

## 📚 **Documentation**

### **Additional Resources**
- **Product Specs**: `/Docs/Product_Specs.md`
- **User Stories**: `/Docs/user_story/`
- **Implementation Guides**: `/Docs/`

### **Support**
- **Issues**: Check existing notebooks for troubleshooting
- **Performance**: Use appropriate profile for your needs
- **Scaling**: Graduate from starter → enterprise as needed

## 🎯 **Key Benefits**

### **For Users**
- ✅ **Clear Learning Path**: Beginner → Professional
- ✅ **Reduced Complexity**: 75% fewer confusing files
- ✅ **Progressive Scaling**: 10 stocks → 1000+ stocks
- ✅ **Faster Setup**: 15-minute quick start

### **For Developers**
- ✅ **Maintainable Code**: Size limits and standards
- ✅ **Version Control**: Jupytext format
- ✅ **Test Separation**: Clean testing structure
- ✅ **Modular Design**: Reusable components

## 🚀 **Next Steps**

1. **Try the quick start**: `examples/quick_start_demo.py`
2. **Choose your path**: Beginner or Professional
3. **Follow the workflow**: Sequential notebook execution
4. **Scale up**: Graduate to enterprise features as needed

---

*This organization was designed following Google TPM best practices for maintainable, scalable data science projects.*

**Last Updated**: January 2025  
**Version**: 2.0 (Post-Refactoring) 