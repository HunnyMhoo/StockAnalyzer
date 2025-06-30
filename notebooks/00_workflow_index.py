# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all,-language_info,-toc,-latex_envs
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🎯 Hong Kong Stock Pattern Recognition - Workflow Index
#
# **Welcome to the complete Hong Kong Stock Pattern Recognition System!**
#
# This notebook serves as your **master navigation hub** for the entire workflow. Choose your path based on your experience level and objectives.

# %%
# Setup and environment validation
from common_setup import setup_notebook, print_setup_summary
from datetime import datetime

print("🚀 Initializing Hong Kong Stock Pattern Recognition System...")
validation = setup_notebook()

if validation and all(validation.values()):
    print("✅ System ready - all components validated!")
else:
    print("⚠️ Some components need attention - check setup below")

# %% [markdown]
# ## 📚 **Workflow Navigation**
#
# ### 🎯 **Quick Start Paths**
#
# | Experience Level | Recommended Path | Time Required |
# |------------------|------------------|---------------|
# | 🔰 **Beginner** | Start with Interactive Demo | 15-30 minutes |
# | 📊 **Data Analyst** | Data Collection → Feature Analysis | 1-2 hours |
# | 🤖 **ML Engineer** | Full Pipeline Workflow | 2-4 hours |
# | 🏢 **Production** | Enterprise Bulk Collection | 4+ hours |

# %% [markdown]
# ## 🔰 **Beginner Track - Quick Demo**
#
# **Perfect for:** First-time users, demos, proof of concept
#
# ### Step 1: Interactive Demo ⭐ **RECOMMENDED START**
# - **Notebook:** `06_interactive_demo.ipynb`
# - **What it does:** Complete end-to-end workflow with interactive widgets
# - **Time:** 15-30 minutes
# - **Features:** 
#   - Interactive parameter selection
#   - Real-time pattern detection
#   - Automatic model training
#   - Professional dashboard with visualizations
#
# **👉 Click here to start:** [Open Interactive Demo](06_interactive_demo.ipynb)

# %% [markdown]
# ## 📊 **Data Analyst Track - Data-Focused**
#
# **Perfect for:** Understanding the data, exploratory analysis
#
# ### Workflow Steps:
# 1. **Data Collection** (`01_data_collection.ipynb`) - 10-15 minutes
#    - Fetch individual stocks for analysis
#    - Data validation and caching
#
# 2. **Basic Bulk Collection** (`02_basic_data_collection.ipynb`) - 20-30 minutes  
#    - Simple, reliable bulk collection (10-50 stocks)
#    - Perfect for beginners with built-in safety limits
#
# 3. **Advanced Bulk Collection** (`02_advanced_data_collection.ipynb`) - 45-90 minutes
#    - Enterprise-grade collection (100-500+ stocks)
#    - Parallel processing and performance optimization
#
# 4. **Feature Analysis** (`04_feature_extraction.ipynb`) - 20-30 minutes
#    - Extract 18+ technical indicators
#    - Pattern feature engineering
#
# 5. **Pattern Visualization** (`07_pattern_match_visualization.ipynb`) - 15-20 minutes
#    - Visualize detected patterns
#    - Chart analysis with overlays

# %% [markdown]
# ## 🤖 **ML Engineer Track - Complete Pipeline**
#
# **Perfect for:** Model development, algorithm tuning, production deployment
#
# ### Full Pipeline Workflow:
# 1. **Data Foundation** - Use Analyst Track Steps 1-3 above
#
# 2. **Model Training** (`05_pattern_model_training.ipynb`) - 30-45 minutes
#    - Train XGBoost and Random Forest models
#    - Cross-validation and performance metrics
#    - Model comparison and selection
#
# 3. **Pattern Scanning** (`06_pattern_scanning.ipynb`) - 20-30 minutes
#    - Apply trained models to detect patterns
#    - Generate confidence scores and rankings
#
# 4. **Signal Analysis** (`08_signal_outcome_tagging.ipynb`) - 30-60 minutes
#    - Tag pattern outcomes (success/failure)
#    - Build feedback loops for model improvement
#    - Performance tracking by confidence bands

# %% [markdown]
# ## 🏢 **Production Track - Enterprise Scale**
#
# **Perfect for:** Large-scale analysis, institutional research, production systems
#
# ### Enterprise Workflow:
# 1. **Bulk Data Collection** (`02_bulk_data_collection_improved.ipynb`)
#    - Configure for 500+ stocks
#    - Parallel processing setup
#    - Enterprise safety controls
#
# 2. **Pattern Model Training** with full dataset
#    - Use complete feature set
#    - Production model validation
#
# 3. **Production Scanning** across entire HK market
#    - Systematic pattern detection
#    - Risk management integration
#
# 4. **Outcome Tracking** and feedback loops
#    - Continuous model improvement
#    - Performance analytics

# %% [markdown]
# ## 🔧 **System Health Check**
#
# Run the cell below to validate your environment and check system status:

# %%
# System health check and environment summary
print("🔍 **System Health Check**")
print("=" * 60)

print_setup_summary()

# Quick validation of key components
print(f"\n🧪 **Component Status:**")

try:
    from stock_analyzer.data import fetch_hk_stocks
    print("   ✅ Data Fetcher - Ready")
except ImportError:
    print("   ❌ Data Fetcher - Import Error")

try:
    from stock_analyzer.features import FeatureExtractor
    print("   ✅ Feature Extractor - Ready")
except ImportError:
    print("   ❌ Feature Extractor - Import Error")

try:
    from stock_analyzer.patterns import PatternScanner
    print("   ✅ Pattern Scanner - Ready")
except ImportError:
    print("   ❌ Pattern Scanner - Import Error")

# Check data availability
import os
data_path = "../data"
if os.path.exists(data_path):
    hk_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and 'HK' in f]
    print(f"   📊 Cached Data Files: {len(hk_files)}")
else:
    print("   📊 No cached data found - run data collection first")

# %% [markdown]
# ## 📖 **Additional Resources**
#
# ### Documentation
# - **[Product Specifications](../Docs/Product_Specs.md)** - Complete system overview
# - **[User Stories](../Docs/user_story/)** - Detailed feature requirements
# - **[Implementation Progress](../progress.md)** - Current development status
#
# ### Development Files
# - **[Notebook Workflow Guide](README_notebook_workflow.md)** - Development best practices
# - **[Conversion Guide](NOTEBOOK_CONVERSION_GUIDE.md)** - Jupytext integration
# - **[Performance Analysis](bulk_data_collection_analysis.md)** - System performance metrics

# %% [markdown]
# ## 🎯 **Getting Started Checklist**
#
# Before diving into the workflows, ensure you have:
#
# ### ✅ **Prerequisites**
# - [ ] Python 3.11+ environment
# - [ ] All dependencies installed (`pip install -r requirements.txt`)
# - [ ] Internet connection for data fetching
# - [ ] ~2GB free space for data cache
#
# ### 🚀 **Quick Start Decision Tree**
#
# **Just want to see it work?** → Use [Interactive Demo](06_interactive_demo.ipynb)
#
# **Need to analyze specific stocks?** → Start with [Data Collection](01_data_collection.ipynb)
#
# **Building ML models?** → Follow the ML Engineer Track above
#
# **Scaling to production?** → Use the Production Track with enterprise settings

# %% [markdown]
# ## 🎓 **Learning Objectives by Track**
#
# ### 🔰 Beginner Track
# - Understand HK stock pattern recognition concept
# - Experience the complete workflow interactively
# - See real pattern detection results
#
# ### 📊 Data Analyst Track  
# - Master HK stock data collection and caching
# - Learn technical indicator feature extraction
# - Understand pattern visualization techniques
#
# ### 🤖 ML Engineer Track
# - Implement pattern recognition ML pipeline
# - Compare model performance and selection
# - Build feedback loops for continuous improvement
#
# ### 🏢 Production Track
# - Scale to enterprise-level data processing
# - Implement production safety and monitoring
# - Build robust, fault-tolerant systems

# %%
print("🎯 **Ready to Begin!**")
print("\nChoose your track above and click the corresponding notebook link.")
print("For immediate results, start with the Interactive Demo.")
print(f"\n📅 Workflow index generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ---
# **🏢 Hong Kong Stock Pattern Recognition System**  
# *Complete workflow navigation hub - choose your path and begin!* 
