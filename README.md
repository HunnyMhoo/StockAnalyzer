# Stock Pattern Recognition Engine

A comprehensive system for Hong Kong stock analysis featuring data collection, pattern labeling, and feature extraction capabilities designed for machine learning model training.

## 🎯 Project Overview

This system provides a complete pipeline for stock pattern recognition and analysis. It includes data collection, manual pattern labeling, feature extraction, and technical analysis capabilities for Hong Kong stocks. The system fetches daily OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance with intelligent local caching, allows manual labeling of chart patterns, and extracts numerical features suitable for machine learning model training.

## ✨ Features

### Data Collection
- **🔄 Intelligent Caching**: Automatic local caching with incremental updates
- **📊 Hong Kong Stock Focus**: Optimized for HK stock ticker formats (e.g., `0700.HK`)
- **📓 Jupyter Notebook Ready**: Designed for interactive data science workflows
- **🚀 Progress Tracking**: Visual progress bars and detailed status updates
- **⚡ Efficient Updates**: Only fetches missing data, not entire datasets
- **🛡️ Error Handling**: Robust error handling with retry logic

### Pattern Labeling
- **🏷️ Manual Pattern Labeling**: Interactive system for defining chart patterns
- **📝 JSON Persistence**: Structured storage with validation and error handling
- **📊 Pattern Visualization**: Optional candlestick chart overlays (with mplfinance)
- **✅ Comprehensive Validation**: Date range, ticker format, and data integrity checks
- **🔄 Pattern Management**: Add, load, and manage pattern collections

### Feature Extraction
- **🔢 18+ Numerical Features**: Comprehensive feature set across 4 categories
- **📈 Technical Indicators**: 15+ indicators including SMA, RSI, MACD, Bollinger Bands
- **⚙️ Configurable Windows**: Customizable analysis periods and parameters
- **📄 ML-Ready Output**: CSV format optimized for machine learning workflows
- **🔍 Data Quality Checks**: Missing value detection and validation

## 🚀 Quick Start

### Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
4. **Open** `notebooks/01_data_collection.ipynb` or `notebooks/04_feature_extraction.ipynb`

### Basic Usage

#### Data Collection
```python
# Import the data fetching functions
from src.data_fetcher import fetch_hk_stocks

# Fetch data for Hong Kong stocks
stock_data = fetch_hk_stocks(
    tickers=['0700.HK', '0005.HK'],  # Tencent, HSBC
    start_date='2023-01-01',
    end_date='2023-12-31',
    force_refresh=False  # Use cache if available
)

# Access individual stock data
tencent_data = stock_data['0700.HK']
print(tencent_data.head())
```

#### Pattern Labeling
```python
# Import pattern labeling functions
from src.pattern_labeler import PatternLabel, PatternLabeler

# Create a pattern label
pattern = PatternLabel(
    ticker="0700.HK",
    start_date="2023-02-10", 
    end_date="2023-03-03",
    label_type="positive",
    notes="Strong upward breakout pattern"
)

# Save to collection
labeler = PatternLabeler()
labeler.add_pattern(pattern)
labeler.save_patterns("labels/my_patterns.json")
```

#### Feature Extraction
```python
# Import feature extraction functions
from src.feature_extractor import extract_features_from_labels

# Extract features from all labeled patterns
features_df = extract_features_from_labels(
    labels_file="labels/labeled_patterns.json",
    output_file="features/labeled_features.csv"
)

print(f"Extracted {len(features_df)} feature sets with {features_df.shape[1]} columns")
```

## 📁 Project Structure

```
StockAnalyze/
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py          # Core data fetching logic
│   ├── pattern_labeler.py       # Manual pattern labeling system
│   ├── pattern_visualizer.py    # Optional chart visualization
│   ├── feature_extractor.py     # Feature extraction for ML
│   └── technical_indicators.py  # Technical analysis indicators
├── notebooks/
│   ├── 01_data_collection.ipynb      # Data collection notebook
│   ├── pattern_labeling_demo.ipynb   # Pattern labeling demo
│   └── 04_feature_extraction.ipynb   # Feature extraction notebook
├── tests/
│   ├── __init__.py
│   ├── test_data_fetcher.py          # Data fetching tests
│   ├── test_pattern_labeler.py       # Pattern labeling tests
│   ├── test_feature_extractor.py     # Feature extraction tests
│   └── test_technical_indicators.py  # Technical indicators tests
├── examples/
│   └── feature_extraction_example.py # Feature extraction examples
├── data/                             # Auto-created cache directory
├── labels/                           # Pattern labels storage
├── features/                         # Extracted features storage
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🔧 API Reference

### Main Functions

#### `fetch_hk_stocks(tickers, start_date, end_date, force_refresh=False)`

Fetch and cache daily OHLCV data for Hong Kong stocks.

**Parameters:**
- `tickers` (List[str]): List of HK stock tickers (e.g., `['0700.HK', '0005.HK']`)
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str): End date in 'YYYY-MM-DD' format  
- `force_refresh` (bool): If True, ignore cache and fetch fresh data

**Returns:**
- `Dict[str, pd.DataFrame]`: Dictionary mapping ticker to DataFrame

#### `validate_tickers(tickers)`

Validate Hong Kong stock ticker formats.

**Parameters:**
- `tickers` (List[str]): List of ticker strings

**Returns:**
- `Tuple[List[str], List[str]]`: (valid_tickers, invalid_tickers)

#### `preview_cached_data(ticker)`

Display a preview of cached data for a specific ticker.

**Parameters:**
- `ticker` (str): Stock ticker to preview

#### `list_cached_tickers()`

Display all cached tickers with their date ranges and record counts.

## 📊 Data Format

Each stock returns a pandas DataFrame with the following columns:

| Column      | Type      | Description                        |
|-------------|-----------|-------------------------------------|
| Date        | datetime  | Trading day (DataFrame index)       |
| Open        | float     | Opening price                       |
| High        | float     | Daily high                          |
| Low         | float     | Daily low                           |
| Close       | float     | Closing price                       |
| Adj Close   | float     | Close adjusted for splits/dividends |
| Volume      | int       | Daily trading volume                |

## 🗂️ Caching System

### How It Works

1. **First Run**: Downloads data from Yahoo Finance and saves to `data/{TICKER}.csv`
2. **Subsequent Runs**: Loads from cache and only fetches missing data
3. **Incremental Updates**: Automatically detects date gaps and fills them
4. **Force Refresh**: Use `force_refresh=True` to ignore cache

### Cache Management

```python
# Check what's in cache
list_cached_tickers()

# Preview specific ticker
preview_cached_data('0700.HK')

# Force refresh all data
fresh_data = fetch_hk_stocks(['0700.HK'], '2023-01-01', '2023-12-31', force_refresh=True)
```

### Cache Location

- **Directory**: `data/` (auto-created)
- **Format**: CSV files named `{TICKER}.csv` (e.g., `0700_HK.csv`)
- **Structure**: Date-indexed with all OHLCV columns

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test class
pytest tests/test_data_fetcher.py::TestTickerValidation
```

### Test Coverage

- **Ticker Validation**: Format validation and error handling
- **Cache Management**: Save/load operations and data integrity
- **Date Range Logic**: Incremental update calculations
- **Yahoo Finance Integration**: API calls with mocking
- **Main Functions**: End-to-end workflow testing
- **Utility Functions**: Notebook helper functions

## 📈 Usage Examples

### Basic Data Collection

```python
# Collect data for major HK stocks
major_stocks = ['0700.HK', '0005.HK', '0941.HK', '0388.HK']
data = fetch_hk_stocks(major_stocks, '2023-01-01', '2023-12-31')

# Quick analysis
for ticker, df in data.items():
    print(f"{ticker}: {len(df)} records, latest price: ${df['Close'].iloc[-1]:.2f}")
```

### Incremental Updates

```python
# Initial data collection
data = fetch_hk_stocks(['0700.HK'], '2023-01-01', '2023-06-30')

# Later, extend the date range (only fetches new data)
extended_data = fetch_hk_stocks(['0700.HK'], '2023-01-01', '2023-12-31')
```

### Data Quality Checks

```python
# Check for missing data
for ticker, df in data.items():
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️ {ticker} has {missing} missing values")
    else:
        print(f"✅ {ticker} data is complete")
```

## ⚙️ Configuration

### Key Constants

```python
DATA_DIR = "data"                    # Cache directory
MAX_RETRIES = 2                      # API retry attempts
HK_TICKER_PATTERN = r"^\d{4}\.HK$"  # Valid ticker format
```

### Supported Ticker Format

- **Valid**: 4-digit number + ".HK" (e.g., `0700.HK`, `0005.HK`)
- **Invalid**: Other formats will be skipped with warnings

## 🔍 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Network Issues**
   - The system retries failed requests automatically
   - Check internet connection for Yahoo Finance access

3. **Invalid Tickers**
   - Use 4-digit format with .HK suffix
   - Check ticker existence on Yahoo Finance

4. **Cache Issues**
   - Delete cache files in `data/` directory to reset
   - Use `force_refresh=True` to bypass cache

### Getting Help

```python
# Validate your tickers first
valid, invalid = validate_tickers(['0700.HK', 'INVALID'])
print(f"Valid: {valid}, Invalid: {invalid}")

# Check cache status
list_cached_tickers()

# Preview specific data
preview_cached_data('0700.HK')
```

## 🔄 Next Steps

This system currently provides data collection, pattern labeling, and feature extraction. Future development includes:

1. **🤖 Model Training**: Train ML models on extracted features
2. **🔍 Pattern Detection**: Automated pattern recognition algorithms
3. **📈 Backtesting**: Validate pattern performance over time
4. **⚡ Real-time Analysis**: Live pattern detection and alerting
5. **🎯 Advanced Features**: Additional technical indicators and pattern types

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is for personal use as part of a stock pattern recognition system.

## 🙏 Acknowledgments

- **Yahoo Finance**: Data source via `yfinance` library
- **Pandas**: Data manipulation and analysis
- **Jupyter**: Interactive development environment

---

**Ready to start collecting Hong Kong stock data for pattern recognition! 🚀** 