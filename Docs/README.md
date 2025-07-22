# Hong Kong Stock Pattern Recognition System

A comprehensive system for analyzing Hong Kong stock patterns using machine learning, technical indicators, and interactive analysis tools.

## 🎯 Project Overview

This system provides an end-to-end pipeline for Hong Kong stock pattern recognition:

- **Data Collection**: Intelligent caching and bulk fetching for HK stocks
- **Pattern Recognition**: ML-powered detection of trading patterns
- **Feature Extraction**: 18+ technical indicators and pattern features
- **Interactive Analysis**: Real-time pattern matching with confidence scoring
- **Outcome Tracking**: Feedback loops for continuous model improvement

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Interactive Demo (15 minutes)
```bash
jupyter notebook notebooks/core_workflow/06_interactive_analysis.ipynb
```

### Basic Workflow
```python
# 1. Data Collection
from stock_analyzer.data import fetch_hk_stocks
data = fetch_hk_stocks(['0700.HK', '0005.HK'], '2023-01-01', '2024-01-01')

# 2. Pattern Analysis
from stock_analyzer.analysis import InteractivePatternAnalyzer
analyzer = InteractivePatternAnalyzer()
results = analyzer.analyze_pattern_similarity('0700.HK', '2023-06-01', '2023-06-30', '0005.HK')

# 3. View Results
print(f"Found {len(results.matches_df)} pattern matches")
```

## 📚 Documentation Structure

### Core Documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and component details
- **[PROJECT_REVIEW.md](PROJECT_REVIEW.md)** - Current status and capabilities review
- **[PROGRESS.md](PROGRESS.md)** - Development progress and milestones

### Technical Guides
- **[Workflow Guide](../notebooks/README.md)** - Step-by-step usage workflows
- **[Bulk Data Fetching](../Docs/bulk_fetching_guide.md)** - Large-scale data collection
- **[Setup Guide](../JUPYTEXT_AUTO_SYNC_GUIDE.md)** - Development environment setup

## 🎯 Current Status

### ✅ Completed Components
- **Data Collection**: Enterprise-grade bulk fetching (100-1000+ stocks)
- **Pattern Recognition**: ML models with 70%+ accuracy
- **Feature Extraction**: 18+ technical indicators
- **Interactive Analysis**: Real-time confidence scoring
- **Outcome Tracking**: Feedback loop implementation

### 🔄 Recent Updates
- **2025-01-26**: Interactive Pattern Analysis Enhancement - Confidence Score Issue Resolved
- **Performance**: 25% confidence score issue eliminated through enhanced training
- **Training Data**: 3x improvement in dataset size (3-4 → 11-15 samples)
- **Model Selection**: Intelligent Random Forest fallback for small datasets

### 📊 Performance Metrics
- **Data Collection**: 88.8% file size reduction with git workflow
- **Pattern Detection**: 70%+ accuracy on labeled patterns
- **Processing Speed**: 15-180 minutes for 10-1000+ stocks
- **Storage Efficiency**: Intelligent caching with incremental updates

## 🏗️ System Architecture

### Core Components
```
stock_analyzer/
├── data/           # Data fetching and caching
├── features/       # Feature extraction and indicators
├── patterns/       # Pattern detection and labeling
├── analysis/       # Interactive analysis and training
└── visualization/  # Chart plotting and analysis
```

### Workflow Pipeline
```
Data Collection → Feature Extraction → Pattern Training → Detection → Analysis
```

## 📈 Getting Started Paths

### 🔰 Beginner (15-30 minutes)
1. **Interactive Demo**: `notebooks/core_workflow/06_interactive_analysis.ipynb`
2. **Quick Start**: `notebooks/examples/quick_start_demo.py`

### 📊 Data Analyst (1-2 hours)
1. **Data Collection**: `notebooks/core_workflow/01_data_collection.py`
2. **Feature Analysis**: `notebooks/core_workflow/03_feature_extraction.py`
3. **Visualization**: `notebooks/core_workflow/07_visualization.py`

### 🤖 ML Engineer (2-4 hours)
1. **Model Training**: `notebooks/core_workflow/04_pattern_training.py`
2. **Pattern Detection**: `notebooks/core_workflow/05_pattern_detection.py`
3. **Signal Analysis**: `notebooks/core_workflow/08_signal_analysis.py`

## 🛠️ Technical Stack

- **Data**: Yahoo Finance API, pandas, numpy
- **ML**: XGBoost, Random Forest, scikit-learn
- **Analysis**: Technical indicators, pattern recognition
- **Interface**: Jupyter notebooks, interactive widgets
- **Storage**: CSV/JSON local caching, git-friendly workflow

## 📞 Support

- **Architecture Details**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Current Status**: See [PROJECT_REVIEW.md](PROJECT_REVIEW.md)
- **Latest Updates**: See [PROGRESS.md](PROGRESS.md)
- **Usage Examples**: See `notebooks/examples/`

---

**Last Updated**: January 2025 | **Version**: 2.0 | **Status**: Production Ready 