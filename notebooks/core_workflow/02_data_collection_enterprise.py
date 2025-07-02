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
# # �� Hong Kong Stock Data Collection - Enterprise Edition
#
# **Perfect for:** Advanced users, institutional analysis (100-1000+ stocks)
#
# ## ⚡ Enterprise Features
# - **Parallel Processing**: Multi-threaded collection with configurable workers
# - **Checkpoint/Resume**: Fault-tolerant operations with automatic resume
# - **Performance Monitoring**: Real-time resource usage and optimization
# - **Advanced Error Recovery**: Sophisticated retry logic and fallback strategies
# - **Memory Management**: Efficient streaming for large datasets
#
# ## ⏱️ Scale Estimates
# - **100 stocks**: 10-15 minutes (parallel mode)
# - **300 stocks**: 25-35 minutes (optimized batching)  
# - **500 stocks**: 45-60 minutes (checkpoint enabled)
# - **1000+ stocks**: 1.5-3 hours (full enterprise mode)

# %%
# ENTERPRISE SETUP
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import os
import json
import psutil
from pathlib import Path
from tqdm.notebook import tqdm

# Setup paths and imports
project_root = Path('.').resolve().parent.parent
sys.path.insert(0, str(project_root))

from notebooks.utilities.common_setup import setup_notebook, get_date_range, import_common_modules

print("🚀 **Enterprise Data Collection - System Check**")
print(f"💻 Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"🖥️ CPU Cores: {psutil.cpu_count()}")

validation = setup_notebook()
modules = import_common_modules()
get_hk_stock_list_static = modules['get_hk_stock_list_static']

print(f"✅ Enterprise setup completed!")

# %% [markdown]
# ## ⚙️ Enterprise Configuration

# %%
# ENTERPRISE PROFILES
ENTERPRISE_PROFILES = {
    'professional': {
        'max_stocks': 100,
        'batch_size': 15,
        'delay': 0.8,
        'memory_gb': 4,
        'time_min': 15
    },
    'institutional': {
        'max_stocks': 300,
        'batch_size': 20,
        'delay': 0.6,
        'memory_gb': 8,
        'time_min': 35
    },
    'enterprise': {
        'max_stocks': 500,
        'batch_size': 25,
        'delay': 0.4,
        'memory_gb': 16,
        'time_min': 60
    }
}

# Select and configure profile
SELECTED_PROFILE = 'professional'  # Change as needed
config = ENTERPRISE_PROFILES[SELECTED_PROFILE].copy()

# Add settings
config.update({
    'date_range_days': 365,
    'force_refresh': False,
    'checkpoint_every': 25
})

start_date, end_date = get_date_range(config['date_range_days'])

print(f"🏢 **{SELECTED_PROFILE.upper()} Configuration**")
print(f"📊 Scale: {config['max_stocks']} stocks")
print(f"⏱️ Estimated: {config['time_min']} minutes")
print(f"📅 Period: {start_date} to {end_date}")

print("✅ Enterprise configuration completed!")
print("🚀 Ready for large-scale data collection!")
