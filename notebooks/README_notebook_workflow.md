# Notebook Workflow Improvements

## Overview
Phase 1 improvements implemented to enhance notebook development and collaboration for the Hong Kong Stock Pattern Recognition project.

## What's New

### 1. Automatic Output Stripping (nbstripout)
**What it does:** Automatically removes all output cells, execution counts, and large embedded images from notebooks when you commit to git.

**Benefits:**
- **Smaller diffs:** No more 3MB JSON blobs in pull requests
- **No merge conflicts:** Output cells won't cause merge conflicts anymore  
- **Reduced repo size:** Binaries/plots only exist in working copy
- **Cleaner history:** Focus on code changes, not output variations

**How it works:**
- Installed automatically via git hooks
- Runs every time you `git commit`
- Notebook code, markdown, and metadata stay intact
- Only outputs are removed from committed version

### 2. Python Script Exports
**What it does:** Key notebooks now have corresponding `.py` files for better code review.

**Available scripts:**
- `01_data_collection.py` - Core data fetching logic
- `04_feature_extraction.py` - Feature engineering algorithms  
- `05_pattern_model_training.py` - ML model training pipeline
- `06_pattern_scanning.py` - Pattern detection logic

**Benefits:**
- **Better code reviews:** Plain text diffs instead of JSON
- **Search & grep:** Find functions and variables easily
- **Git blame:** Track changes to specific algorithms
- **Documentation:** Clear view of the core logic

## Developer Workflow

### Daily Use
1. **Develop normally** in Jupyter notebooks
2. **Commit as usual** - nbstripout handles output stripping automatically
3. **Review code** using the `.py` files for algorithm changes
4. **Update `.py` exports** when making significant changes to core notebooks

### Updating Python Exports
When you make significant changes to the core notebooks, refresh the `.py` exports:

```bash
# Export specific notebook
jupyter nbconvert --to script notebooks/04_feature_extraction.ipynb --output-dir notebooks/

# Or export all key notebooks
jupyter nbconvert --to script notebooks/{01_data_collection,04_feature_extraction,05_pattern_model_training,06_pattern_scanning}.ipynb --output-dir notebooks/
```

## File Size Impact
The improvement in repository efficiency:

| Notebook | Original Size | Python Script | Reduction |
|----------|---------------|---------------|-----------|
| `01_data_collection.ipynb` | 216KB | 4.8KB | 98% |
| `04_feature_extraction.ipynb` | 40KB | 8.3KB | 79% |  
| `05_pattern_model_training.ipynb` | 153KB | 10.4KB | 93% |
| `06_pattern_scanning.ipynb` | 39KB | 11KB | 72% |

## Technical Details

### nbstripout Configuration
- **Git filter:** Configured in `.gitattributes`
- **Automatic:** Runs on `git add` and `git commit`
- **Reversible:** Original outputs preserved in working directory

### Dependencies
- `nbstripout>=0.6.1` added to `requirements.txt`
- No additional manual setup required for new team members

## Next Steps (Future Phases)
- **Phase 2:** Automated Jupytext sync for bidirectional editing
- **Phase 3:** CI/CD pipeline with Papermill for notebook testing

---

*This workflow improves code review quality for our Hong Kong stock pattern recognition algorithms while maintaining the rich notebook development experience.*

# Notebook Workflow Documentation

This directory contains Jupyter notebooks that demonstrate the complete StockAnalyzer workflow for Hong Kong stock pattern recognition.

## 📚 Available Notebooks

### Core Analysis Workflow
1. **[01_data_collection.ipynb](01_data_collection.ipynb)** - Data fetching and caching for HK stocks
2. **[02_bulk_data_collection_improved.ipynb](02_bulk_data_collection_improved.ipynb)** - Bulk data collection optimization
3. **[04_feature_extraction.ipynb](04_feature_extraction.ipynb)** - Technical indicator feature extraction
4. **[05_pattern_model_training.ipynb](05_pattern_model_training.ipynb)** - ML model training and evaluation
5. **[06_pattern_scanning.ipynb](06_pattern_scanning.ipynb)** - Pattern detection and scanning

### Interactive Demo
6. **[06_interactive_demo.ipynb](06_interactive_demo.ipynb)** ⭐ **NEW** - Complete end-to-end interactive workflow

### Visualization & Analysis
7. **[07_pattern_match_visualization.ipynb](07_pattern_match_visualization.ipynb)** - Pattern visualization
8. **[08_signal_outcome_tagging.ipynb](08_signal_outcome_tagging.ipynb)** - Signal outcome analysis

## 🎯 Interactive Demo Features

The **06_interactive_demo.ipynb** notebook provides a comprehensive, interactive demonstration of the entire StockAnalyzer workflow:

### ✅ Key Features:
- **Papermill-compatible** for CI automation
- **Interactive widgets** for real-time parameter adjustment
- **Pattern detection** using "bull trend → dip → false resistance break" logic
- **Feature extraction** with 18 specialized technical indicators
- **Model training** using GradientBoostingClassifier
- **Interactive visualizations** with Plotly charts
- **Professional dashboard** with KPIs and performance metrics
- **Run Again functionality** for workflow re-execution

### 🚀 Quick Start:
```bash
# Open the interactive demo
jupyter notebook 06_interactive_demo.ipynb

# Or run with Papermill for CI
papermill 06_interactive_demo.ipynb output.ipynb -p QUICK_MODE True
```

### 📊 Workflow Steps:
1. **Parameter Configuration** - Configure analysis parameters via widgets
2. **Ticker Tagging** - Detect preferred patterns in selected stocks
3. **Feature Extraction** - Extract technical indicators for ML training
4. **Model Training** - Train pattern detection classifier
5. **Interactive Dashboard** - View results with charts and KPIs

### 🎛️ Interactive Controls:
- **Ticker Selection**: Multi-select from HK blue chips
- **Lookback Window**: 30-730 days (slider)
- **Model Threshold**: 0.50-0.95 confidence (slider)
- **Quick Mode**: Enable for CI smoke testing

### 📈 Outputs:
- `data/tagged_tickers.csv` - Pattern detection results
- `data/features/interactive_demo_features.csv` - Extracted features
- `models/latest.joblib` - Trained pattern detection model
- Interactive charts and performance dashboards

## 🔧 Technical Requirements

All notebooks require:
- Python 3.11+
- Libraries: pandas ≥ 2.0, yfinance, scikit-learn, plotly, ipywidgets, joblib, tqdm
- StockAnalyzer package (local src/ directory)

### Additional for Interactive Demo:
- ipywidgets for interactive controls
- plotly for visualization
- Papermill for CI integration

## 🏗️ Development Workflow

### Running Individual Notebooks:
```bash
cd notebooks/
jupyter notebook <notebook_name>.ipynb
```

### Running via Papermill (CI):
```bash
# Interactive demo with custom parameters
papermill 06_interactive_demo.ipynb output.ipynb \
  -p SELECTED_TICKERS '["0700.HK", "0005.HK"]' \
  -p QUICK_MODE True \
  -p MODEL_THRESHOLD 0.75
```

### Paired .py Files:
The interactive demo includes a paired `.py` file for version control and CI integration using Jupytext.

## 🧪 Testing

Unit tests are available for the interactive demo functionality:
```bash
python -m pytest tests/test_interactive_demo.py -v
```

## 📚 Documentation

For detailed information about the Hong Kong stock pattern recognition engine, see:
- [Product Specifications](../Docs/Product_Specs.md)
- [User Stories](../Docs/user_story/)
- [Implementation Progress](../progress.md)

---

**🎓 Built for Hong Kong Stock Pattern Recognition**  
*StockAnalyzer Interactive Demo - Complete end-to-end workflow* 