# System Architecture

## 🏗️ Overview

The Hong Kong Stock Pattern Recognition System is built with a modular architecture that separates concerns across data collection, feature extraction, pattern recognition, and analysis components.

## 📁 Component Architecture

### Core Module Structure
```
stock_analyzer/
├── __init__.py              # Package exports and API
├── config.py                # Centralized configuration
├── data/                    # Data collection and management
│   ├── fetcher.py          # Yahoo Finance data fetching
│   ├── universe.py         # HK stock universe definitions
│   └── cache.py            # Intelligent caching system
├── features/                # Feature extraction pipeline
│   ├── extractor.py        # Main feature extraction class
│   └── indicators.py       # Technical indicators library
├── patterns/                # Pattern detection and labeling
│   ├── scanner.py          # Pattern scanning engine
│   └── labeler.py          # Manual pattern labeling
├── analysis/                # Analysis and training components
│   ├── interactive.py      # Interactive pattern analysis
│   ├── trainer.py          # Model training pipeline
│   ├── evaluator.py        # Model evaluation metrics
│   ├── outcome.py          # Signal outcome tracking
│   └── quality.py          # Data quality analysis
├── visualization/           # Chart and visualization
│   └── charts.py           # Pattern visualization tools
└── utils/                   # Utilities and helpers
    ├── notebook.py         # Notebook integration
    └── widgets.py          # Interactive widgets
```

## 🔄 Data Flow Pipeline

### 1. Data Collection Layer
```
Yahoo Finance API → Data Fetcher → Local Cache → Data Validation
```

**Key Components:**
- **Data Fetcher** (`data/fetcher.py`): Handles API calls with rate limiting
- **Cache Manager**: Intelligent incremental updates
- **Validation Engine**: Data quality checks and error handling

### 2. Feature Extraction Layer
```
Cached Data → Technical Indicators → Feature Windows → ML Features
```

**Key Components:**
- **Feature Extractor** (`features/extractor.py`): Main extraction pipeline
- **Technical Indicators** (`features/indicators.py`): 18+ indicators
- **Feature Windows**: Sliding window analysis

### 3. Pattern Recognition Layer
```
Labeled Patterns → Model Training → Pattern Scanner → Confidence Scoring
```

**Key Components:**
- **Pattern Scanner** (`patterns/scanner.py`): Core detection engine
- **Model Trainer** (`analysis/trainer.py`): ML model training
- **Pattern Labeler** (`patterns/labeler.py`): Manual labeling system

### 4. Analysis Layer
```
Pattern Matches → Interactive Analysis → Outcome Tracking → Feedback Loop
```

**Key Components:**
- **Interactive Analyzer** (`analysis/interactive.py`): Real-time analysis
- **Outcome Tracker** (`analysis/outcome.py`): Performance feedback
- **Quality Analyzer** (`analysis/quality.py`): Data quality assessment

## 🧩 Core Components Detail

### Data Collection (`stock_analyzer/data/`)

#### `fetcher.py` - Data Fetching Engine
```python
class DataFetcher:
    - fetch_hk_stocks()         # Main fetching function
    - _get_cache_filename()     # Cache file management
    - _load_cached_data()       # Cache retrieval
    - validate_tickers()        # Ticker validation
```

**Features:**
- Yahoo Finance API integration
- Intelligent caching with incremental updates
- Rate limiting and error handling
- HK stock format validation (XXXX.HK)

#### `universe.py` - Stock Universe Management
```python
Functions:
    - get_hk_stock_list_static()    # Curated major stocks
    - get_hk_stocks_by_sector()     # Sector-based selection
    - get_top_hk_stocks()           # Top N stocks
```

### Feature Extraction (`stock_analyzer/features/`)

#### `extractor.py` - Feature Extraction Pipeline
```python
class FeatureExtractor:
    - extract_features()            # Main extraction method
    - _extract_trend_context()      # Trend analysis features
    - _extract_correction_phase()   # Correction pattern features
    - _extract_technical_indicators() # Technical indicator features
```

**Feature Categories:**
1. **Trend Context** (6 features): SMA trends, momentum, volatility
2. **Correction Phase** (4 features): Drawdown, recovery patterns
3. **False Support Break** (4 features): Support levels, breakout detection
4. **Technical Indicators** (6+ features): RSI, MACD, volume analysis

#### `indicators.py` - Technical Indicators Library
```python
Functions:
    - simple_moving_average()       # SMA calculation
    - relative_strength_index()     # RSI calculation
    - macd()                        # MACD calculation
    - bollinger_bands()             # Bollinger bands
    - find_support_resistance_levels() # Support/resistance detection
    - calculate_drawdown_metrics()  # Drawdown analysis
```

### Pattern Recognition (`stock_analyzer/patterns/`)

#### `scanner.py` - Pattern Scanning Engine
```python
class PatternScanner:
    - scan_for_patterns()           # Main scanning function
    - _apply_sliding_window()       # Window-based analysis
    - _calculate_confidence()       # Confidence scoring
    - _rank_matches()               # Result ranking
```

**Scanning Process:**
1. Load trained ML model
2. Apply sliding window across stock data
3. Extract features for each window
4. Generate confidence scores
5. Rank and filter results

#### `labeler.py` - Pattern Labeling System
```python
class PatternLabeler:
    - add_pattern()                 # Add labeled pattern
    - save_patterns()               # Save to JSON
    - load_patterns()               # Load from JSON
    - validate_pattern()            # Pattern validation
```

### Analysis Components (`stock_analyzer/analysis/`)

#### `interactive.py` - Interactive Pattern Analysis
```python
class InteractivePatternAnalyzer:
    - analyze_pattern_similarity()  # Main analysis workflow
    - _train_temporary_model()      # Dynamic model training
    - _scan_for_matches()           # Pattern matching
    - _process_results()            # Result processing
```

**Analysis Workflow:**
1. Extract features from positive example
2. Generate negative examples
3. Train temporary model
4. Scan market for similar patterns
5. Return confidence-ranked results

#### `trainer.py` - Model Training Pipeline
```python
class ModelTrainer:
    - train_model()                 # Main training function
    - _prepare_features()           # Feature preparation
    - _train_xgboost()              # XGBoost training
    - _train_random_forest()        # Random Forest training
    - _evaluate_model()             # Model evaluation
```

## 🔧 Technical Implementation

### Configuration Management (`config.py`)
```python
class Settings:
    - DATA_DIR                      # Data directory path
    - ALERT_THRESHOLD               # Pattern confidence threshold
    - SMA_WINDOWS                   # SMA window periods
    - DEFAULT_WINDOW_SIZE           # Pattern window size
```

### Error Handling Strategy
- **Custom Exceptions**: Specific error types for each component
- **Graceful Degradation**: Fallback mechanisms for failures
- **Logging**: Comprehensive error tracking and debugging

### Performance Optimizations
- **Vectorized Operations**: Pandas/NumPy optimizations
- **Intelligent Caching**: Incremental updates and file management
- **Batch Processing**: Efficient bulk operations
- **Memory Management**: Efficient data structures

## 📊 Data Models

### Pattern Label Structure
```python
@dataclass
class PatternLabel:
    ticker: str                     # Stock ticker (e.g., "0700.HK")
    start_date: str                 # Pattern start date
    end_date: str                   # Pattern end date
    label_type: str                 # "positive", "negative", "neutral"
    notes: str                      # Additional notes
```

### Feature Window Structure
```python
@dataclass
class FeatureWindow:
    ticker: str                     # Stock ticker
    window_start_date: str          # Window start date
    window_end_date: str            # Window end date
    features: Dict[str, float]      # Feature values
    label: Optional[str]            # Pattern label (if labeled)
```

### Analysis Result Structure
```python
@dataclass
class PatternAnalysisResult:
    matches_df: pd.DataFrame        # Matched patterns with confidence
    scanning_summary: Dict          # Scanning statistics
    analysis_metadata: Dict         # Analysis parameters
    success: bool                   # Analysis success status
```

## 🚀 Deployment Architecture

### Development Environment
- **Jupyter Notebooks**: Interactive development and analysis
- **Jupytext Integration**: Git-friendly notebook management
- **Local Caching**: Fast development iteration

### Production Considerations
- **Scalable Data Collection**: Batch processing with rate limiting
- **Model Versioning**: Timestamped model artifacts
- **Result Storage**: Structured output for downstream systems
- **Monitoring**: Performance and error tracking

## 🔍 Integration Points

### External Dependencies
- **Yahoo Finance API**: Primary data source
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting models
- **Matplotlib/Plotly**: Visualization

### Internal Module Dependencies
```
analysis → patterns → features → data
    ↓         ↓         ↓        ↓
visualization ← utils ← config ← base
```

## 📈 Extensibility

### Adding New Features
1. Implement in `features/indicators.py`
2. Integrate in `features/extractor.py`
3. Update configuration in `config.py`

### Adding New Models
1. Implement in `analysis/trainer.py`
2. Update `patterns/scanner.py` for inference
3. Add evaluation metrics in `analysis/evaluator.py`

### Adding New Data Sources
1. Implement fetcher in `data/fetcher.py`
2. Update caching logic
3. Add validation rules

---

**Architecture Version**: 2.0 | **Last Updated**: January 2025 