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
# <span style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">An Exception was encountered at '<a href="#papermill-error-cell">In [1]</a>'.</span>

# %% [markdown]
# <span id="papermill-error-cell" style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">Execution using papermill encountered an exception here and stopped:</span>

# %%
# Hong Kong Stock Data Collection

This notebook demonstrates how to fetch and cache daily OHLCV data for Hong Kong stocks using Yahoo Finance. The data will be used for chart pattern recognition and model training.

## Features
- ✅ Fetch data from Yahoo Finance
- ✅ Intelligent local caching
- ✅ Incremental updates
- ✅ Progress tracking
- ✅ Error handling and recovery
- ✅ Data validation and preview


# %%
## Setup and Imports

First, let's import the necessary libraries and our data fetching module.


# %%
# Add the src directory to Python path
import sys
import os
sys.path.append(os.path.join('..', 'src'))

# Import our data fetching functions
from data_fetcher import (
    fetch_hk_stocks,
    validate_tickers,
    preview_cached_data,
    list_cached_tickers
)

# Standard data science libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8')

print("✅ All imports successful!")
print(f"📅 Current date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# %%
## Step 1: Define Target Stocks

Let's define some popular Hong Kong stocks to fetch data for. These are commonly traded stocks that should have good data availability.


# %%
# Define target Hong Kong stocks
target_tickers = [
    '6969.HK',  
]

# Validate ticker formats
valid_tickers, invalid_tickers = validate_tickers(target_tickers)

print(f"📊 Target stocks: {len(target_tickers)} total")
print(f"✅ Valid tickers: {len(valid_tickers)} - {valid_tickers}")
if invalid_tickers:
    print(f"❌ Invalid tickers: {len(invalid_tickers)} - {invalid_tickers}")

# Display stock names for reference


print("\n🏢 Stock names:")
for ticker in valid_tickers:
    print(f"  {ticker}: {stock_names.get(ticker, 'Unknown')}")


# %%
valid_tickers

# %%
## Step 2: Set Date Range

Define the date range for historical data. We'll fetch approximately 1 year of data for pattern recognition training.


# %%
# Define date range (1 year of data)
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

print(f"📅 Date range for data collection:")
print(f"   Start: {start_date}")
print(f"   End: {end_date}")
print(f"   Duration: ~365 days")

# Calculate expected trading days (approximately 250 trading days per year)
total_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
expected_trading_days = int(total_days * 5/7 * 0.95)  # Rough estimate excluding weekends and holidays

print(f"   Expected trading days: ~{expected_trading_days}")


# %%
## Step 3: Check Existing Cache

Before fetching new data, let's see what's already cached locally.


# %%
# Check existing cached data
print("📁 Current cache status:")
list_cached_tickers()


# %%
## Step 4: Fetch Stock Data

Now let's fetch the data. This will use intelligent caching - if data already exists, it will only fetch missing data.


# %%
# Fetch stock data with caching
print("🚀 Starting data collection...\n")

stock_data = fetch_hk_stocks(
    tickers=valid_tickers,
    start_date=start_date,
    end_date=end_date,
    force_refresh=True  # Set to True to ignore cache and fetch fresh data
)

print(f"\n✅ Data collection completed!")
print(f"📊 Successfully fetched data for {len(stock_data)} stocks")


# %%
## Step 5: Data Validation and Preview

Let's validate the fetched data and get a preview of what we have.


# %%
# Validate data structure and content
print("🔍 Data validation:")
print("="*50)

for ticker, data in stock_data.items():
    print(f"\n📈 {ticker} ({stock_names.get(ticker, 'Unknown')})")
    print(f"   📅 Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"   📊 Records: {len(data)}")
    print(f"   📋 Columns: {list(data.columns)}")
    print(f"   💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"   📊 Average volume: {data['Volume'].mean():,.0f}")
    
    # Check for missing data
    missing_data = data.isnull().sum().sum()
    if missing_data > 0:
        print(f"   ⚠️  Missing data points: {missing_data}")
    else:
        print(f"   ✅ No missing data")
    
    # Display recent data sample
    print(f"   📋 Recent data:")
    print(data.tail(3).round(2))

print(f"\n📊 Overall statistics:")
print(f"   Total records: {sum(len(data) for data in stock_data.values()):,}")
print(f"   Average records per stock: {sum(len(data) for data in stock_data.values()) / len(stock_data):.0f}")
print(f"\n✅ Data validation completed!")

