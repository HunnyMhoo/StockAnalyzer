# Stock Pattern Recognition Engine - Data Collection Module

A Jupyter notebook-friendly data fetching and caching system for Hong Kong stocks, designed to support chart pattern recognition and machine learning model training.

## 🎯 Project Overview

This module provides the foundational data layer for a personal stock pattern recognition system. It fetches daily OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance with intelligent local caching to minimize API calls and improve performance.

## ✨ Features

- **🔄 Intelligent Caching**: Automatic local caching with incremental updates
- **📊 Hong Kong Stock Focus**: Optimized for HK stock ticker formats (e.g., `0700.HK`)
- **📓 Jupyter Notebook Ready**: Designed for interactive data science workflows
- **🚀 Progress Tracking**: Visual progress bars and detailed status updates
- **⚡ Efficient Updates**: Only fetches missing data, not entire datasets
- **🛡️ Error Handling**: Robust error handling with retry logic
- **🔍 Data Validation**: Automatic data quality checks and validation
- **📈 Rich Output**: Formatted displays with emojis and clear status messages

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
4. **Open** `notebooks/01_data_collection.ipynb`

### Basic Usage

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

## 📁 Project Structure

```
StockAnalyze/
├── src/
│   ├── __init__.py
│   └── data_fetcher.py          # Core data fetching logic
├── notebooks/
│   └── 01_data_collection.ipynb # Main data collection notebook
├── tests/
│   ├── __init__.py
│   └── test_data_fetcher.py     # Comprehensive unit tests
├── data/                        # Auto-created cache directory
├── requirements.txt             # Python dependencies
└── README.md                    # This file
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

This data collection module provides the foundation for:

1. **📊 Pattern Labeling**: Manually label chart patterns on fetched data
2. **🔧 Feature Engineering**: Extract technical indicators and features  
3. **🤖 Model Training**: Train ML models on labeled patterns
4. **🔍 Pattern Detection**: Scan new data for similar patterns
5. **📈 Backtesting**: Validate pattern performance over time

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