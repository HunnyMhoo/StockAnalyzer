# 🏢 Complete Guide: Fetching All Hong Kong Stocks

This guide explains how to fetch data for **all Hong Kong stocks** using our Stock Pattern Recognition Engine.

## 🎯 Quick Start

```python
# Basic example: Fetch top 50 HK stocks
from stock_analyzer.data import get_top_hk_stocks, fetch_hk_stocks_bulk
from datetime import datetime, timedelta

end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

stock_data = fetch_top_50_hk_stocks(
    start_date=start_date,
    end_date=end_date,
    batch_size=10,
    delay_between_batches=2.0
)

print(f"Fetched {len(stock_data)} stocks")
```

## 📊 Available Approaches

### 1. **Curated Major Stocks** (Recommended)
- **Function**: `get_hk_stock_list_static()`
- **Count**: ~34 major stocks
- **Reliability**: ⭐⭐⭐⭐⭐
- **Speed**: Fast
- **Coverage**: Blue chips, major tech, finance, property

```python
from hk_stock_universe import get_hk_stock_list_static
stocks = get_hk_stock_list_static()
```

### 2. **Sector-Based Fetching**
- **Function**: `get_hk_stocks_by_sector(sector)`
- **Sectors**: `tech_stocks`, `finance`, `property`, `blue_chips`
- **Reliability**: ⭐⭐⭐⭐⭐

```python
from hk_stock_universe import get_hk_stocks_by_sector

tech_stocks = get_hk_stocks_by_sector('tech_stocks')      # 10 stocks
finance_stocks = get_hk_stocks_by_sector('finance')       # 10 stocks  
property_stocks = get_hk_stocks_by_sector('property')     # 10 stocks
blue_chips = get_hk_stocks_by_sector('blue_chips')        # 15 stocks
```

### 3. **Top N Stocks**
- **Function**: `get_top_hk_stocks(n)`
- **Range**: 1-50 stocks
- **Sorted by**: Market cap/popularity

```python
from hk_stock_universe import get_top_hk_stocks

top_10 = get_top_hk_stocks(10)
top_25 = get_top_hk_stocks(25)
top_50 = get_top_hk_stocks(50)
```

### 4. **Comprehensive Discovery**
- **Function**: `get_comprehensive_hk_stock_list()`
- **Features**: Validation, filtering, expansion
- **Max tickers**: Configurable

```python
from hk_stock_universe import get_comprehensive_hk_stock_list

stock_info = get_comprehensive_hk_stock_list(
    include_major=True,
    validate_tickers=True,
    max_tickers=100,
    min_market_cap=1e9  # 1B HKD minimum
)

print(f"Found {len(stock_info['valid_stocks'])} valid stocks")
```

## ⚡ Bulk Fetching Strategies

### Strategy 1: Conservative (Recommended for Beginners)
```python
from stock_analyzer.data import fetch_hk_stocks_bulk

stock_data = fetch_all_major_hk_stocks(
    start_date='2023-01-01',
    end_date='2024-01-01',
    max_stocks=50,
    batch_size=10,
    delay_between_batches=3.0
)
```
- **Time**: 15-30 minutes
- **Success Rate**: 95%+
- **Best for**: Learning, testing

### Strategy 2: Balanced (Production Ready)
```python
stock_data = fetch_hk_stocks_bulk(
    tickers=get_top_hk_stocks(100),
    start_date=start_date,
    end_date=end_date,
    batch_size=20,
    delay_between_batches=2.0,
    skip_failed=True
)
```
- **Time**: 30-60 minutes
- **Success Rate**: 90%+
- **Best for**: Regular analysis

### Strategy 3: Comprehensive (Enterprise)
```python
# For maximum coverage
all_stocks = get_comprehensive_hk_stock_list(max_tickers=500)
stock_data = fetch_hk_stocks_bulk(
    tickers=all_stocks['valid_stocks'],
    start_date=start_date,
    end_date=end_date,
    batch_size=25,
    delay_between_batches=2.0,
    max_retries=3
)
```
- **Time**: 2-4 hours
- **Success Rate**: 85%+
- **Best for**: Research, backtesting

## 🛠️ Convenience Functions

### Pre-built Sector Functions
```python
from bulk_data_fetcher import (
    fetch_hk_tech_stocks,
    fetch_hk_finance_stocks,
    fetch_hk_property_stocks
)

# Fetch by sector with one line
tech_data = fetch_hk_tech_stocks('2023-01-01', '2024-01-01')
finance_data = fetch_hk_finance_stocks('2023-01-01', '2024-01-01')
property_data = fetch_hk_property_stocks('2023-01-01', '2024-01-01')
```

### Top Stocks Function
```python
from bulk_data_fetcher import fetch_top_50_hk_stocks

# One-liner for top 50 stocks
top_50_data = fetch_top_50_hk_stocks(
    start_date='2023-01-01',
    end_date='2024-01-01'
)
```

## 📈 Data Management

### Save Bulk Data
```python
from bulk_data_fetcher import save_bulk_data

save_bulk_data(
    stock_data=stock_data,
    base_dir="all_hk_stocks",
    create_summary=True
)
```

### Create Summary Reports
```python
from bulk_data_fetcher import create_bulk_fetch_summary

summary_df = create_bulk_fetch_summary(stock_data)
print(summary_df.head())

# Key metrics per stock:
# - Records count
# - Date range
# - Latest close price
# - Price range (min/max)
# - Volume statistics
# - Data quality score
```

## ⚠️ Important Considerations

### Rate Limiting
- **Yahoo Finance**: ~200 requests/hour
- **Batch size**: 10-25 stocks per batch
- **Delay**: 2-5 seconds between batches
- **Retry strategy**: 2-3 attempts per failed stock

### Error Handling
```python
stock_data = fetch_hk_stocks_bulk(
    tickers=my_stocks,
    start_date=start_date,
    end_date=end_date,
    skip_failed=True,      # Continue on failures
    max_retries=2,         # Retry failed stocks
    force_refresh=False    # Use cache when possible
)
```

### Performance Tips
1. **Use caching**: Don't set `force_refresh=True` unless needed
2. **Start small**: Test with 10-20 stocks first
3. **Monitor progress**: Watch the console output
4. **Run overnight**: For 100+ stocks
5. **Check network**: Stable internet connection required

## 📊 Stock Universe Coverage

### Major Categories
| Category | Count | Examples |
|----------|-------|----------|
| **Blue Chips** | 15 | Tencent (0700), HSBC (0005), China Mobile (0941) |
| **Tech Stocks** | 10 | Alibaba (9988), Meituan (3690), Xiaomi (1810) |
| **Finance** | 10 | ICBC (1398), Ping An (2318), AIA (1299) |
| **Property** | 10 | Sun Hung Kai (0016), New World (0017) |

### Stock Code Ranges
- **0001-0999**: Traditional stocks
- **1000-9999**: Newer listings
- **9000+**: Recent IPOs (ADRs, tech companies)

## 🚀 Advanced Usage

### Parallel Processing (Use with Caution)
```python
from bulk_data_fetcher import fetch_hk_stocks_parallel

# WARNING: May hit rate limits
parallel_data = fetch_hk_stocks_parallel(
    tickers=small_stock_list,
    start_date=start_date,
    end_date=end_date,
    max_workers=3  # Keep low
)
```

### Custom Stock Lists
```python
# Create your own stock list
my_hk_stocks = [
    '0700.HK',  # Tencent
    '0005.HK',  # HSBC
    '9988.HK',  # Alibaba
    '3690.HK',  # Meituan
    '1299.HK'   # AIA
]

custom_data = fetch_hk_stocks_bulk(
    tickers=my_hk_stocks,
    start_date=start_date,
    end_date=end_date
)
```

### Market Cap Filtering
```python
from hk_stock_universe import get_hk_stocks_by_market_cap

# Only stocks with market cap > 10B HKD
large_caps = get_hk_stocks_by_market_cap(min_market_cap=10e9)
```

## 📓 Jupyter Notebook Usage

Use `notebooks/02_bulk_data_collection.ipynb` for interactive exploration:

```python
# In Jupyter notebook
%load_ext autoreload
%autoreload 2

from bulk_data_fetcher import *
from hk_stock_universe import *

# Interactive fetching with progress bars
stock_data = fetch_top_50_hk_stocks(
    start_date='2023-01-01',
    end_date='2024-01-01',
    batch_size=10
)
```

## 🔧 Troubleshooting

### Common Issues
1. **Rate limits**: Increase delays, reduce batch size
2. **Network timeouts**: Increase retry count
3. **Invalid tickers**: Use validation functions
4. **Memory issues**: Process in smaller chunks

### Error Messages
- `"Failed to fetch"`: Network/API issue, retry later
- `"Invalid ticker"`: Check ticker format (XXXX.HK)
- `"No data"`: Stock may be delisted or suspended

## 📚 Next Steps

After fetching bulk data:
1. **Data validation**: Check for missing data
2. **Pattern recognition**: Apply ML algorithms
3. **Portfolio optimization**: Select best stocks
4. **Backtesting**: Test trading strategies

## 💡 Pro Tips

1. **Start with major stocks** for reliability
2. **Use sector-based approach** for focused analysis  
3. **Cache aggressively** to avoid re-fetching
4. **Monitor API quotas** to avoid blocks
5. **Run large jobs overnight** for best results
6. **Validate data quality** before analysis

---

This completes our guide to fetching all Hong Kong stocks. The system is designed to be flexible, robust, and respectful of API limits while providing comprehensive market coverage. 