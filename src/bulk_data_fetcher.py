"""
Bulk Data Fetcher for Hong Kong Stocks

This module provides optimized functions for fetching data for large numbers
of Hong Kong stocks with proper batching, rate limiting, and error handling.
"""

import os
import sys
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm.notebook import tqdm
import warnings


def nb_print(*args, **kwargs):
    """Notebook-friendly print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()

from data_fetcher import fetch_hk_stocks, validate_tickers
from hk_stock_universe import (
    get_comprehensive_hk_stock_list,
    get_top_hk_stocks,
    get_hk_stocks_by_sector,
    save_stock_list,
    load_stock_list
)


def fetch_hk_stocks_bulk(
    tickers: List[str],
    start_date: str,
    end_date: str,
    batch_size: int = 20,
    delay_between_batches: float = 2.0,
    max_retries: int = 2,
    force_refresh: bool = False,
    skip_failed: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch stock data for a large number of HK stocks in batches.
    
    This function optimizes bulk data fetching by:
    - Processing stocks in smaller batches
    - Adding delays between batches to respect rate limits
    - Handling failures gracefully
    - Providing detailed progress tracking
    
    Args:
        tickers: List of HK stock tickers
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        batch_size: Number of stocks to process per batch
        delay_between_batches: Delay in seconds between batches
        max_retries: Number of retries for failed stocks
        force_refresh: If True, ignore cache and fetch fresh data
        skip_failed: If True, continue processing even if some stocks fail
        
    Returns:
        Dict[str, pd.DataFrame]: Successfully fetched stock data
    """
    nb_print(f"🚀 Starting bulk fetch for {len(tickers)} HK stocks")
    nb_print(f"⚙️  Batch size: {batch_size}, Delay: {delay_between_batches}s")
    nb_print(f"📅 Date range: {start_date} to {end_date}")
    
    # Validate all tickers first
    valid_tickers, invalid_tickers = validate_tickers(tickers)
    
    if invalid_tickers:
        nb_print(f"⚠️  Skipping {len(invalid_tickers)} invalid tickers: {invalid_tickers[:5]}{'...' if len(invalid_tickers) > 5 else ''}")
    
    if not valid_tickers:
        nb_print("❌ No valid tickers to process")
        return {}
    
    all_stock_data = {}
    failed_stocks = []
    
    # Process in batches
    total_batches = (len(valid_tickers) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(valid_tickers))
        batch_tickers = valid_tickers[start_idx:end_idx]
        
        nb_print(f"\n📦 Processing batch {batch_num + 1}/{total_batches}: {len(batch_tickers)} stocks")
        nb_print(f"   Tickers: {', '.join(batch_tickers)}")
        
        retry_count = 0
        batch_success = False
        
        while retry_count <= max_retries and not batch_success:
            try:
                # Fetch data for this batch
                batch_data = fetch_hk_stocks(
                    tickers=batch_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    force_refresh=force_refresh
                )
                
                # Add successful fetches to result
                for ticker, data in batch_data.items():
                    if data is not None and len(data) > 0:
                        all_stock_data[ticker] = data
                    else:
                        failed_stocks.append(ticker)
                
                batch_success = True
                nb_print(f"   ✅ Batch completed: {len(batch_data)} stocks fetched")
                
            except Exception as e:
                retry_count += 1
                nb_print(f"   ⚠️  Batch failed (attempt {retry_count}/{max_retries + 1}): {e}")
                
                if retry_count <= max_retries:
                    nb_print(f"   🔄 Retrying in {delay_between_batches}s...")
                    time.sleep(delay_between_batches)
                else:
                    nb_print(f"   ❌ Batch failed after {max_retries + 1} attempts")
                    if not skip_failed:
                        raise
                    failed_stocks.extend(batch_tickers)
        
        # Rate limiting between batches
        if batch_num < total_batches - 1:  # Don't delay after last batch
            nb_print(f"   ⏳ Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)
    
    # Final summary
    nb_print(f"\n🎉 Bulk fetch completed!")
    nb_print(f"   ✅ Successfully fetched: {len(all_stock_data)} stocks")
    nb_print(f"   ❌ Failed: {len(failed_stocks)} stocks")
    
    if failed_stocks:
        nb_print(f"   Failed stocks: {', '.join(failed_stocks)}")
    
    success_rate = len(all_stock_data) / len(valid_tickers) * 100
    nb_print(f"   📊 Success rate: {success_rate:.1f}%")
    
    return all_stock_data


def fetch_all_major_hk_stocks(
    start_date: str,
    end_date: str,
    sector: Optional[str] = None,
    max_stocks: Optional[int] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all major Hong Kong stocks.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        sector: Specific sector to fetch (e.g., 'tech_stocks', 'finance', 'property')
        max_stocks: Maximum number of stocks to fetch
        **kwargs: Additional arguments passed to fetch_hk_stocks_bulk
        
    Returns:
        Dict[str, pd.DataFrame]: Stock data
    """
    print(f"🏢 Fetching all major HK stocks...")
    
    if sector:
        tickers = get_hk_stocks_by_sector(sector)
        print(f"📊 Selected sector: {sector}")
    else:
        stock_info = get_comprehensive_hk_stock_list(
            include_major=True,
            validate_tickers=False  # We'll validate in bulk fetch
        )
        tickers = stock_info['major_stocks']
        print(f"📈 Using all major stocks")
    
    if max_stocks:
        tickers = tickers[:max_stocks]
        print(f"🎯 Limited to {max_stocks} stocks")
    
    print(f"📋 Total stocks to fetch: {len(tickers)}")
    
    return fetch_hk_stocks_bulk(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


def fetch_hk_stocks_parallel(
    tickers: List[str],
    start_date: str,
    end_date: str,
    max_workers: int = 5,
    force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Fetch stock data using parallel processing.
    
    WARNING: Use with caution as this may overwhelm the Yahoo Finance API.
    Only recommended for small numbers of stocks or with significant delays.
    
    Args:
        tickers: List of HK stock tickers
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        max_workers: Maximum number of parallel workers
        force_refresh: If True, ignore cache and fetch fresh data
        
    Returns:
        Dict[str, pd.DataFrame]: Stock data
    """
    print(f"⚡ Fetching {len(tickers)} stocks in parallel (max_workers={max_workers})")
    print("⚠️  WARNING: Parallel fetching may hit rate limits")
    
    all_stock_data = {}
    failed_stocks = []
    
    def fetch_single_stock(ticker):
        """Fetch data for a single stock."""
        try:
            result = fetch_hk_stocks(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh
            )
            return ticker, result.get(ticker), None
        except Exception as e:
            return ticker, None, str(e)
    
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {executor.submit(fetch_single_stock, ticker): ticker for ticker in tickers}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_ticker), total=len(tickers), desc="Fetching stocks"):
            ticker, data, error = future.result()
            
            if error:
                failed_stocks.append(ticker)
                print(f"❌ Failed {ticker}: {error}")
            elif data is not None and len(data) > 0:
                all_stock_data[ticker] = data
            else:
                failed_stocks.append(ticker)
    
    print(f"\n✅ Parallel fetch completed: {len(all_stock_data)} succeeded, {len(failed_stocks)} failed")
    return all_stock_data


def create_bulk_fetch_summary(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary DataFrame for bulk-fetched stock data.
    
    Args:
        stock_data: Dictionary of stock data from bulk fetch
        
    Returns:
        pd.DataFrame: Summary of all fetched stocks
    """
    summary_data = []
    
    for ticker, data in stock_data.items():
        summary_data.append({
            'Ticker': ticker,
            'Records': len(data),
            'Start_Date': data.index.min().date(),
            'End_Date': data.index.max().date(),
            'Latest_Close': data['Close'].iloc[-1],
            'Min_Price': data['Close'].min(),
            'Max_Price': data['Close'].max(),
            'Avg_Volume': data['Volume'].mean(),
            'Total_Volume': data['Volume'].sum(),
            'Missing_Data': data.isnull().sum().sum(),
            'Data_Quality': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if not summary_df.empty:
        # Sort by market cap proxy (latest close price)
        summary_df = summary_df.sort_values('Latest_Close', ascending=False)
        summary_df = summary_df.reset_index(drop=True)
    
    return summary_df


def save_bulk_data(
    stock_data: Dict[str, pd.DataFrame],
    base_dir: str = "bulk_data",
    create_summary: bool = True
) -> None:
    """
    Save bulk-fetched stock data to individual CSV files.
    
    Args:
        stock_data: Dictionary of stock data
        base_dir: Base directory to save files
        create_summary: Whether to create a summary file
    """
    print(f"💾 Saving {len(stock_data)} stocks to {base_dir}/")
    
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Save individual stock files
    for ticker, data in tqdm(stock_data.items(), desc="Saving files"):
        filename = f"{ticker.replace('.', '_')}.csv"
        filepath = os.path.join(base_dir, filename)
        data.to_csv(filepath)
    
    # Create and save summary
    if create_summary:
        summary_df = create_bulk_fetch_summary(stock_data)
        summary_path = os.path.join(base_dir, "bulk_fetch_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"📊 Summary saved to {summary_path}")
    
    print(f"✅ All files saved to {base_dir}/")


# Convenience functions for common use cases
def fetch_top_50_hk_stocks(start_date: str, end_date: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Fetch data for top 50 HK stocks."""
    tickers = get_top_hk_stocks(50)
    return fetch_hk_stocks_bulk(tickers, start_date, end_date, **kwargs)


def fetch_hk_tech_stocks(start_date: str, end_date: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Fetch data for HK tech stocks."""
    return fetch_all_major_hk_stocks(start_date, end_date, sector='tech_stocks', **kwargs)


def fetch_hk_finance_stocks(start_date: str, end_date: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Fetch data for HK financial stocks."""
    return fetch_all_major_hk_stocks(start_date, end_date, sector='finance', **kwargs)


def fetch_hk_property_stocks(start_date: str, end_date: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Fetch data for HK property stocks."""
    return fetch_all_major_hk_stocks(start_date, end_date, sector='property', **kwargs)


if __name__ == "__main__":
    # Demo usage
    print("🚀 Bulk Data Fetcher Demo")
    
    # Example: Fetch top 10 stocks
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"📅 Fetching last 30 days: {start_date} to {end_date}")
    
    # Get top 10 for demo
    demo_data = fetch_top_50_hk_stocks(
        start_date=start_date,
        end_date=end_date,
        max_stocks=10,
        batch_size=5
    )
    
    print(f"✅ Demo completed: {len(demo_data)} stocks fetched") 