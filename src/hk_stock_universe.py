"""
Hong Kong Stock Universe Manager

This module provides functionality to fetch and manage the complete list of 
Hong Kong stocks for bulk data collection operations.
"""

import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple
import yfinance as yf
from tqdm.notebook import tqdm
import time
import warnings


# Major HK stocks by category for quick access
MAJOR_HK_STOCKS = {
    'blue_chips': [
        '0700.HK',  # Tencent Holdings
        '0005.HK',  # HSBC Holdings
        '0941.HK',  # China Mobile
        '0388.HK',  # Hong Kong Exchanges
        '1299.HK',  # AIA Group
        '2318.HK',  # Ping An Insurance
        '1398.HK',  # ICBC
        '0939.HK',  # China Construction Bank
        '3988.HK',  # Bank of China
        '2388.HK',  # BOC Hong Kong
        '0823.HK',  # Link REIT
        '0883.HK',  # CNOOC
        '0016.HK',  # Sun Hung Kai Properties
        '0017.HK',  # New World Development
        '1113.HK',  # CK Asset Holdings
    ],
    'tech_stocks': [
        '0700.HK',  # Tencent
        '9988.HK',  # Alibaba
        '3690.HK',  # Meituan
        '1024.HK',  # Kuaishou
        '9618.HK',  # JD.com
        '9999.HK',  # NetEase
        '0981.HK',  # SMIC
        '2015.HK',  # Li Auto
        '9868.HK',  # Xpeng
        '1810.HK',  # Xiaomi
    ],
    'finance': [
        '0005.HK',  # HSBC
        '1398.HK',  # ICBC
        '0939.HK',  # CCB
        '3988.HK',  # BOC
        '2388.HK',  # BOC HK
        '2318.HK',  # Ping An
        '1299.HK',  # AIA
        '1359.HK',  # China Cinda
        '6818.HK',  # CEB Bank
        '1988.HK',  # Minsheng Bank
    ],
    'property': [
        '0016.HK',  # Sun Hung Kai
        '0017.HK',  # New World
        '1113.HK',  # CK Asset
        '1109.HK',  # China Resources Land
        '0688.HK',  # China Overseas
        '0012.HK',  # Henderson Land
        '0101.HK',  # Hang Lung Properties
        '0083.HK',  # Sino Land
        '0019.HK',  # Swire Pacific A
        '1997.HK',  # Wharf Real Estate
    ]
}


def get_hk_stock_list_static() -> List[str]:
    """
    Get a curated list of major Hong Kong stocks.
    
    This is the most reliable method as it uses a manually curated list
    of liquid, actively traded HK stocks.
    
    Returns:
        List[str]: List of HK stock tickers
    """
    all_stocks = []
    for category, stocks in MAJOR_HK_STOCKS.items():
        all_stocks.extend(stocks)
    
    # Remove duplicates while preserving order
    unique_stocks = []
    seen = set()
    for stock in all_stocks:
        if stock not in seen:
            unique_stocks.append(stock)
            seen.add(stock)
    
    return unique_stocks


def get_hse_stocks_by_range(start_code: int = 1, end_code: int = 9999) -> List[str]:
    """
    Generate HK stock tickers by code range.
    
    Hong Kong stocks use 4-digit codes from 0001 to 9999.
    This generates potential tickers in the range.
    
    Args:
        start_code: Starting stock code (default: 1)
        end_code: Ending stock code (default: 9999)
        
    Returns:
        List[str]: List of potential HK stock tickers
    """
    tickers = []
    for code in range(start_code, end_code + 1):
        ticker = f"{code:04d}.HK"
        tickers.append(ticker)
    
    return tickers


def validate_hk_tickers_batch(tickers: List[str], 
                            batch_size: int = 50,
                            delay: float = 1.0) -> Tuple[List[str], List[str]]:
    """
    Validate a batch of HK tickers by checking if they exist on Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols to validate
        batch_size: Number of tickers to process in each batch
        delay: Delay between batches in seconds
        
    Returns:
        Tuple of (valid_tickers, invalid_tickers)
    """
    valid_tickers = []
    invalid_tickers = []
    
    print(f"🔍 Validating {len(tickers)} HK tickers...")
    
    # Process in batches to avoid overwhelming the API
    for i in tqdm(range(0, len(tickers), batch_size), desc="Validating batches"):
        batch = tickers[i:i + batch_size]
        
        for ticker in batch:
            try:
                # Quick check - try to get basic info
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Check if we got valid data
                if info and len(info) > 1:  # Valid stocks have substantial info
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
                    
            except Exception:
                invalid_tickers.append(ticker)
        
        # Rate limiting
        if delay > 0 and i + batch_size < len(tickers):
            time.sleep(delay)
    
    print(f"✅ Validation complete: {len(valid_tickers)} valid, {len(invalid_tickers)} invalid")
    return valid_tickers, invalid_tickers


def get_hk_stocks_by_market_cap(min_market_cap: float = 1e9) -> List[str]:
    """
    Get HK stocks filtered by minimum market cap.
    
    Args:
        min_market_cap: Minimum market cap in HKD (default: 1 billion)
        
    Returns:
        List[str]: List of HK stocks meeting market cap criteria
    """
    major_stocks = get_hk_stock_list_static()
    filtered_stocks = []
    
    print(f"📊 Filtering stocks by market cap >= {min_market_cap/1e9:.1f}B HKD...")
    
    for ticker in tqdm(major_stocks, desc="Checking market caps"):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            market_cap = info.get('marketCap', 0)
            if market_cap >= min_market_cap:
                filtered_stocks.append(ticker)
                
        except Exception as e:
            warnings.warn(f"Could not get market cap for {ticker}: {e}")
            continue
    
    print(f"✅ Found {len(filtered_stocks)} stocks meeting criteria")
    return filtered_stocks


def get_comprehensive_hk_stock_list(
    include_major: bool = True,
    validate_tickers: bool = True,
    max_tickers: Optional[int] = None,
    min_market_cap: Optional[float] = None
) -> Dict[str, List[str]]:
    """
    Get a comprehensive list of Hong Kong stocks with various options.
    
    Args:
        include_major: Include curated major stocks
        validate_tickers: Validate tickers exist on Yahoo Finance
        max_tickers: Maximum number of tickers to return
        min_market_cap: Minimum market cap filter
        
    Returns:
        Dict with categorized stock lists
    """
    result = {
        'major_stocks': [],
        'all_stocks': [],
        'valid_stocks': [],
        'summary': {}
    }
    
    # Start with major stocks if requested
    if include_major:
        result['major_stocks'] = get_hk_stock_list_static()
        print(f"📈 Loaded {len(result['major_stocks'])} major HK stocks")
    
    # Get comprehensive list (for demonstration, using major + some range)
    comprehensive_list = result['major_stocks'].copy()
    
    # Optionally add more stocks by code range (be careful with this)
    if max_tickers and max_tickers > len(comprehensive_list):
        print(f"🔍 Expanding list to {max_tickers} stocks...")
        # Add some popular ranges
        additional_stocks = []
        
        # Add some known active ranges
        for code in [1, 2, 3, 4, 6, 8, 11, 12, 17, 19, 27, 66, 83, 101, 175, 267, 388, 700, 823, 883, 939, 941, 992, 1024, 1038, 1109, 1113, 1177, 1299, 1359, 1398, 1810, 1876, 1918, 1997, 2007, 2018, 2313, 2318, 2388, 2628, 3328, 3988, 6030, 6862, 9618, 9888, 9988, 9999]:
            ticker = f"{code:04d}.HK"
            if ticker not in comprehensive_list:
                additional_stocks.append(ticker)
        
        comprehensive_list.extend(additional_stocks[:max_tickers - len(comprehensive_list)])
    
    result['all_stocks'] = comprehensive_list
    
    # Apply market cap filter if specified
    if min_market_cap:
        filtered_stocks = []
        for ticker in comprehensive_list:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', 0)
                if market_cap >= min_market_cap:
                    filtered_stocks.append(ticker)
            except:
                continue
        comprehensive_list = filtered_stocks
    
    # Validate tickers if requested
    if validate_tickers:
        valid, invalid = validate_hk_tickers_batch(comprehensive_list)
        result['valid_stocks'] = valid
        result['invalid_stocks'] = invalid
    else:
        result['valid_stocks'] = comprehensive_list
        result['invalid_stocks'] = []
    
    # Create summary
    result['summary'] = {
        'total_major': len(result['major_stocks']),
        'total_all': len(result['all_stocks']),
        'total_valid': len(result['valid_stocks']),
        'total_invalid': len(result['invalid_stocks']),
        'validation_rate': len(result['valid_stocks']) / len(result['all_stocks']) * 100 if result['all_stocks'] else 0
    }
    
    return result


def save_stock_list(stock_list: List[str], filename: str = "hk_stocks.txt") -> None:
    """
    Save stock list to a text file.
    
    Args:
        stock_list: List of stock tickers
        filename: Output filename
    """
    with open(filename, 'w') as f:
        for ticker in stock_list:
            f.write(f"{ticker}\n")
    
    print(f"💾 Saved {len(stock_list)} stocks to {filename}")


def load_stock_list(filename: str = "hk_stocks.txt") -> List[str]:
    """
    Load stock list from a text file.
    
    Args:
        filename: Input filename
        
    Returns:
        List[str]: List of stock tickers
    """
    try:
        with open(filename, 'r') as f:
            stocks = [line.strip() for line in f if line.strip()]
        
        print(f"📁 Loaded {len(stocks)} stocks from {filename}")
        return stocks
        
    except FileNotFoundError:
        print(f"❌ File {filename} not found")
        return []


# Quick access functions
def get_top_hk_stocks(n: int = 50) -> List[str]:
    """Get top N HK stocks by popularity/liquidity."""
    major_stocks = get_hk_stock_list_static()
    return major_stocks[:n]


def get_hk_stocks_by_sector(sector: str) -> List[str]:
    """Get HK stocks by sector."""
    if sector.lower() in MAJOR_HK_STOCKS:
        return MAJOR_HK_STOCKS[sector.lower()]
    else:
        available_sectors = list(MAJOR_HK_STOCKS.keys())
        print(f"❌ Sector '{sector}' not found. Available: {available_sectors}")
        return []


if __name__ == "__main__":
    # Demo usage
    print("🚀 HK Stock Universe Demo")
    
    # Get major stocks
    major = get_hk_stock_list_static()
    print(f"📈 Major stocks: {len(major)}")
    
    # Get stocks by sector
    tech = get_hk_stocks_by_sector('tech_stocks')
    print(f"💻 Tech stocks: {len(tech)}")
    
    print("✅ Demo complete!") 