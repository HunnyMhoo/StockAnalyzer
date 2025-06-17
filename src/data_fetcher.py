"""
Stock Data Fetcher for Hong Kong Stocks

This module provides notebook-friendly functions to fetch and cache daily OHLCV data
for Hong Kong stocks using Yahoo Finance API with intelligent caching.
"""

import os
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from tqdm.notebook import tqdm


def nb_print(*args, **kwargs):
    """Notebook-friendly print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


# Configuration constants
DATA_DIR = "data"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
HK_TICKER_PATTERN = r"^\d{4}\.HK$"
MAX_RETRIES = 2
RETRY_DELAY = 1.0


def _validate_hk_ticker(ticker: str) -> bool:
    """
    Validate Hong Kong stock ticker format.
    
    Args:
        ticker: Stock ticker string
        
    Returns:
        bool: True if valid HK ticker format
    """
    return bool(re.match(HK_TICKER_PATTERN, ticker))


def _ensure_data_directory() -> None:
    """Create data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✓ Created data directory: {DATA_DIR}/")


def _get_cache_filename(ticker: str) -> str:
    """
    Get the cache filename for a ticker.
    
    Args:
        ticker: Stock ticker (e.g., '0700.HK')
        
    Returns:
        str: Cache filename
    """
    # Sanitize ticker for filename (replace . with _)
    safe_ticker = ticker.replace('.', '_')
    return os.path.join(DATA_DIR, f"{safe_ticker}.csv")


def _load_cached_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load cached data for a ticker if it exists.
    
    Args:
        ticker: Stock ticker
        
    Returns:
        DataFrame or None: Cached data if exists, None otherwise
    """
    cache_file = _get_cache_filename(ticker)
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Ensure timezone-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        warnings.warn(f"Failed to load cached data for {ticker}: {e}")
        return None


def _save_data_to_cache(ticker: str, data: pd.DataFrame) -> None:
    """
    Save data to cache file.
    
    Args:
        ticker: Stock ticker
        data: DataFrame to save
    """
    cache_file = _get_cache_filename(ticker)
    
    try:
        # Ensure timezone-naive before saving
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        data.to_csv(cache_file)
        print(f"  💾 Saved {len(data)} records to cache")
    except Exception as e:
        warnings.warn(f"Failed to save data for {ticker}: {e}")


def _fetch_from_yahoo(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch data from Yahoo Finance with retry logic.
    
    Args:
        ticker: Stock ticker
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame or None: Fetched data or None if failed
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Use Ticker.history() method for better compatibility
            # This method is more reliable than yf.download() for single stocks
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                return None
            
            # Ensure timezone-naive
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Validate required columns (Note: Adj Close might not be present with auto_adjust=True)
            required_basic = ["Open", "High", "Low", "Close", "Volume"]
            missing_cols = set(required_basic) - set(data.columns)
            if missing_cols:
                warnings.warn(f"Missing columns for {ticker}: {missing_cols}")
                return None
            
            # Add Adj Close if not present (for compatibility)
            if "Adj Close" not in data.columns:
                data["Adj Close"] = data["Close"]
            
            return data
            
        except Exception as e:
            if attempt < MAX_RETRIES:
                nb_print(f"  ⚠️  Attempt {attempt + 1} failed, retrying...")
                time.sleep(RETRY_DELAY)  # Add delay between retries
                continue
            else:
                warnings.warn(f"Failed to fetch {ticker} after {MAX_RETRIES + 1} attempts: {e}")
                return None
    
    return None


def _determine_date_range(cached_data: Optional[pd.DataFrame], 
                         start_date: str, 
                         end_date: str) -> Tuple[str, str, bool]:
    """
    Determine what date range to fetch based on cached data.
    
    Args:
        cached_data: Existing cached data or None
        start_date: Requested start date
        end_date: Requested end date
        
    Returns:
        Tuple of (fetch_start, fetch_end, needs_fetch)
    """
    if cached_data is None:
        return start_date, end_date, True
    
    cached_start = cached_data.index.min().strftime('%Y-%m-%d')
    cached_end = cached_data.index.max().strftime('%Y-%m-%d')
    
    # Check if we need to fetch new data
    requested_start = datetime.strptime(start_date, '%Y-%m-%d')
    requested_end = datetime.strptime(end_date, '%Y-%m-%d')
    cached_end_date = datetime.strptime(cached_end, '%Y-%m-%d')
    
    # If cache covers the requested range, no fetch needed
    if (datetime.strptime(cached_start, '%Y-%m-%d') <= requested_start and 
        cached_end_date >= requested_end):
        return start_date, end_date, False
    
    # If we need more recent data, fetch from day after cached end
    if cached_end_date < requested_end:
        next_day = (cached_end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        return next_day, end_date, True
    
    # If we need older data, fetch from requested start to cached start
    if datetime.strptime(cached_start, '%Y-%m-%d') > requested_start:
        day_before = (datetime.strptime(cached_start, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        return start_date, day_before, True
    
    return start_date, end_date, False


def _merge_with_cache(ticker: str, 
                     new_data: pd.DataFrame, 
                     cached_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge new data with cached data.
    
    Args:
        ticker: Stock ticker
        new_data: Newly fetched data
        cached_data: Existing cached data
        
    Returns:
        DataFrame: Merged data
    """
    if cached_data is None:
        return new_data
    
    # Combine and remove duplicates
    combined = pd.concat([cached_data, new_data])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    
    return combined


def validate_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate ticker formats and return valid/invalid lists.
    
    Args:
        tickers: List of ticker strings
        
    Returns:
        Tuple of (valid_tickers, invalid_tickers)
    """
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        if _validate_hk_ticker(ticker):
            valid_tickers.append(ticker)
        else:
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers


def preview_cached_data(ticker: str) -> None:
    """
    Display a preview of cached data for a ticker.
    
    Args:
        ticker: Stock ticker to preview
    """
    if not _validate_hk_ticker(ticker):
        print(f"❌ Invalid ticker format: {ticker}")
        return
    
    cached_data = _load_cached_data(ticker)
    
    if cached_data is None:
        print(f"📁 No cached data found for {ticker}")
        return
    
    print(f"📊 Cached data for {ticker}:")
    print(f"   📅 Date range: {cached_data.index.min().date()} to {cached_data.index.max().date()}")
    print(f"   📈 Records: {len(cached_data)}")
    print(f"   💰 Price range: ${cached_data['Close'].min():.2f} - ${cached_data['Close'].max():.2f}")
    print("\n📋 Recent data:")
    print(cached_data.tail(3).round(2))


def list_cached_tickers() -> None:
    """Display all cached tickers with their date ranges."""
    _ensure_data_directory()
    
    cache_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if not cache_files:
        print("📁 No cached data found")
        return
    
    print("📊 Cached tickers:")
    print("-" * 60)
    
    for cache_file in sorted(cache_files):
        ticker = cache_file.replace('_', '.').replace('.csv', '')
        cached_data = _load_cached_data(ticker)
        
        if cached_data is not None:
            start_date = cached_data.index.min().date()
            end_date = cached_data.index.max().date()
            record_count = len(cached_data)
            
            print(f"📈 {ticker:8} | {start_date} to {end_date} | {record_count:4} records")


def fetch_hk_stocks(tickers: List[str], 
                   start_date: str, 
                   end_date: str, 
                   force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Fetch and cache daily OHLCV data for Hong Kong stocks.
    
    This is the main function for fetching stock data with intelligent caching.
    Data is cached locally to minimize API calls and improve performance.
    
    Args:
        tickers: List of HK stock tickers (e.g., ['0700.HK', '0005.HK'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        force_refresh: If True, ignore cache and fetch fresh data
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping ticker to DataFrame
        
    Example:
        >>> data = fetch_hk_stocks(['0700.HK'], '2023-01-01', '2023-12-31')
        >>> tencent_data = data['0700.HK']
        >>> print(tencent_data.head())
    """
    nb_print(f"🚀 Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Ensure data directory exists
    _ensure_data_directory()
    
    # Validate tickers
    valid_tickers, invalid_tickers = validate_tickers(tickers)
    
    if invalid_tickers:
        nb_print(f"⚠️  Invalid ticker formats (skipping): {invalid_tickers}")
    
    if not valid_tickers:
        nb_print("❌ No valid tickers to process")
        return {}
    
    results = {}
    
    # Process each ticker with progress bar
    for ticker in tqdm(valid_tickers, desc="Processing tickers"):
        nb_print(f"\n📊 Processing {ticker}...")
        
        try:
            # Load cached data
            cached_data = None if force_refresh else _load_cached_data(ticker)
            
            if cached_data is not None and not force_refresh:
                nb_print(f"  📁 Found cached data: {len(cached_data)} records")
                
                # Determine if we need to fetch additional data
                fetch_start, fetch_end, needs_fetch = _determine_date_range(
                    cached_data, start_date, end_date
                )
                
                if needs_fetch:
                    nb_print(f"  🔄 Fetching additional data: {fetch_start} to {fetch_end}")
                    new_data = _fetch_from_yahoo(ticker, fetch_start, fetch_end)
                    
                    if new_data is not None:
                        # Merge with cached data
                        merged_data = _merge_with_cache(ticker, new_data, cached_data)
                        _save_data_to_cache(ticker, merged_data)
                        results[ticker] = merged_data
                    else:
                        nb_print(f"  ⚠️  Failed to fetch additional data, using cached data only")
                        results[ticker] = cached_data
                else:
                    nb_print(f"  ✅ Cache covers requested range")
                    results[ticker] = cached_data
            else:
                if force_refresh:
                    nb_print(f"  🔄 Force refresh enabled, fetching fresh data...")
                else:
                    nb_print(f"  📡 No cached data, fetching from Yahoo Finance...")
                
                # Fetch fresh data
                data = _fetch_from_yahoo(ticker, start_date, end_date)
                
                if data is not None:
                    _save_data_to_cache(ticker, data)
                    results[ticker] = data
                else:
                    nb_print(f"  ❌ Failed to fetch data for {ticker}")
                    continue
            
            # Filter data to requested date range
            if ticker in results:
                mask = (results[ticker].index >= pd.to_datetime(start_date)) & \
                       (results[ticker].index <= pd.to_datetime(end_date))
                results[ticker] = results[ticker].loc[mask]
                
                nb_print(f"  ✅ Final dataset: {len(results[ticker])} records")
                
        except Exception as e:
            nb_print(f"  ❌ Error processing {ticker}: {e}")
            continue
    
    nb_print(f"\n🎉 Successfully processed {len(results)} out of {len(valid_tickers)} tickers")
    
    # Display summary
    if results:
        nb_print("\n📈 Summary:")
        for ticker, data in results.items():
            nb_print(f"  {ticker}: {len(data)} records, "
                  f"${data['Close'].iloc[-1]:.2f} (latest close)")
    
    return results 