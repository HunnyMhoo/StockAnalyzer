#!/usr/bin/env python3
"""
YFinance Compatibility Test Script

This script tests yfinance compatibility and helps diagnose data structure issues.
"""

import sys
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_yfinance_basic():
    """Test basic yfinance functionality."""
    print("🧪 Testing yfinance basic functionality...")
    print(f"📦 yfinance version: {yf.__version__}")
    
    ticker = "0700.HK"  # Tencent
    
    try:
        # Test basic download
        print(f"\n🔍 Testing basic download for {ticker}...")
        stock = yf.Ticker(ticker)
        
        # Test info
        print("📊 Testing stock info...")
        info = stock.info
        print(f"   Info keys: {len(info)} items")
        if 'longName' in info:
            print(f"   Company: {info.get('longName', 'Unknown')}")
        
        # Test history with different parameters
        print("\n📈 Testing history download...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Try different download methods
        methods = [
            {"name": "Method 1: Basic", "params": {"period": "5d"}},
            {"name": "Method 2: Date range", "params": {"start": start_date, "end": end_date}},
            {"name": "Method 3: With options", "params": {
                "start": start_date, 
                "end": end_date,
                "auto_adjust": True,
                "prepost": True,
                "threads": True,
                "progress": False
            }},
        ]
        
        for method in methods:
            try:
                print(f"\n   🔄 {method['name']}...")
                data = stock.history(**method['params'])
                
                if data is not None and not data.empty:
                    print(f"      ✅ Success: {len(data)} records")
                    print(f"      📋 Columns: {list(data.columns)}")
                    print(f"      📅 Date range: {data.index.min()} to {data.index.max()}")
                    print(f"      💰 Sample close: ${data['Close'].iloc[-1]:.2f}")
                    
                    # Check for required columns
                    required = ["Open", "High", "Low", "Close", "Volume"]
                    missing = [col for col in required if col not in data.columns]
                    if missing:
                        print(f"      ⚠️  Missing required columns: {missing}")
                    else:
                        print(f"      ✅ All required columns present")
                        
                    return data  # Return successful data for further testing
                    
                else:
                    print(f"      ❌ No data returned")
                    
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return None

def test_yfinance_download_function():
    """Test the yf.download function directly."""
    print("\n🧪 Testing yf.download function directly...")
    
    ticker = "0700.HK"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    print(f"📅 Date range: {start_date} to {end_date}")
    
    download_configs = [
        {"name": "Minimal", "params": {
            "tickers": ticker,
            "start": start_date,
            "end": end_date,
            "progress": False
        }},
        {"name": "With auto_adjust", "params": {
            "tickers": ticker,
            "start": start_date,
            "end": end_date,
            "progress": False,
            "auto_adjust": True
        }},
        {"name": "Full config", "params": {
            "tickers": ticker,
            "start": start_date,
            "end": end_date,
            "progress": False,
            "auto_adjust": True,
            "prepost": True,
            "threads": True
        }},
    ]
    
    for config in download_configs:
        try:
            print(f"\n   🔄 {config['name']}...")
            data = yf.download(**config['params'])
            
            if data is not None and not data.empty:
                print(f"      ✅ Success: {len(data)} records")
                print(f"      📋 Columns: {list(data.columns)}")
                print(f"      🔍 Data shape: {data.shape}")
                print(f"      📊 Sample data:")
                print(data.tail(2).round(2))
                return data
            else:
                print(f"      ❌ No data returned")
                
        except Exception as e:
            print(f"      ❌ Failed: {e}")

def test_alternative_tickers():
    """Test with different HK tickers to see if it's ticker-specific."""
    print("\n🧪 Testing alternative HK tickers...")
    
    tickers = [
        "0005.HK",  # HSBC
        "0941.HK",  # China Mobile
        "0388.HK",  # HK Exchange
        "^HSI",     # Hang Seng Index
    ]
    
    for ticker in tickers:
        print(f"\n   📊 Testing {ticker}...")
        try:
            data = yf.download(
                ticker,
                period="5d",
                progress=False,
                auto_adjust=True
            )
            
            if data is not None and not data.empty:
                print(f"      ✅ Success: {len(data)} records")
                print(f"      📋 Columns: {list(data.columns)}")
            else:
                print(f"      ❌ No data")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")

def main():
    """Run all compatibility tests."""
    print("🚀 YFinance Compatibility Test Suite")
    print("=" * 60)
    
    # Test 1: Basic functionality
    sample_data = test_yfinance_basic()
    
    # Test 2: Direct download function
    if sample_data is None:
        sample_data = test_yfinance_download_function()
    
    # Test 3: Alternative tickers
    test_alternative_tickers()
    
    print("\n" + "=" * 60)
    if sample_data is not None:
        print("🎉 YFinance is working! Data structure confirmed.")
        print("💡 Use the successful configuration in your code.")
    else:
        print("❌ YFinance compatibility issues detected.")
        print("💡 Check your internet connection and yfinance version.")
    
    # Show recommendations
    print("\n📝 Recommendations:")
    print("1. Update yfinance: pip install --upgrade yfinance")
    print("2. Use auto_adjust=True to avoid warnings")
    print("3. Use stock.history() method instead of yf.download() for single stocks")
    print("4. Check ticker format (e.g., '0700.HK' for Hong Kong stocks)")

if __name__ == "__main__":
    main() 