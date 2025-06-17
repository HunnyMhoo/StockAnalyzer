#!/usr/bin/env python3
"""
Comprehensive Examples: How to Fetch All Hong Kong Stocks

This script demonstrates various approaches to fetch HK stock data in bulk,
from small curated lists to comprehensive market coverage.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, timedelta
from hk_stock_universe import *
from bulk_data_fetcher import *
from data_fetcher import fetch_hk_stocks


def example_1_sector_based_fetching():
    """Example 1: Fetch stocks by sector."""
    print("\n" + "="*70)
    print("📊 EXAMPLE 1: Sector-Based Fetching")
    print("="*70)
    
    # Set date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"📅 Date range: {start_date} to {end_date}")
    
    # Fetch different sectors
    sectors = ['tech_stocks', 'finance', 'property']
    
    for sector in sectors:
        print(f"\n🏢 Fetching {sector.upper()} stocks...")
        stocks = get_hk_stocks_by_sector(sector)
        print(f"   📋 Found {len(stocks)} stocks: {stocks}")
        
        # For demo, just show the process (don't actually fetch to save time)
        print(f"   ⚙️  Would fetch data for {len(stocks)} stocks")


def example_2_top_stocks():
    """Example 2: Fetch top N stocks."""
    print("\n" + "="*70)
    print("📈 EXAMPLE 2: Top Stocks Fetching")
    print("="*70)
    
    # Get top stocks by different criteria
    top_10 = get_top_hk_stocks(10)
    top_25 = get_top_hk_stocks(25)
    top_50 = get_top_hk_stocks(50)
    
    print(f"🎯 Top 10 stocks: {len(top_10)} stocks")
    print(f"   {top_10}")
    
    print(f"\n🎯 Top 25 stocks: {len(top_25)} stocks")
    print(f"   First 10: {top_25[:10]}")
    
    print(f"\n🎯 Top 50 stocks: {len(top_50)} stocks")
    print(f"   First 10: {top_50[:10]}")


def example_3_comprehensive_list():
    """Example 3: Get comprehensive stock list."""
    print("\n" + "="*70)
    print("🔍 EXAMPLE 3: Comprehensive Stock Discovery")
    print("="*70)
    
    # Get comprehensive stock information
    stock_info = get_comprehensive_hk_stock_list(
        include_major=True,
        validate_tickers=False,  # Skip validation for demo
        max_tickers=100
    )
    
    print("📊 Stock List Summary:")
    for key, value in stock_info['summary'].items():
        print(f"   {key}: {value}")
    
    print(f"\n📋 Major stocks sample: {stock_info['major_stocks'][:10]}")
    print(f"📋 All stocks sample: {stock_info['all_stocks'][:10]}")


def example_4_bulk_fetching_demo():
    """Example 4: Actual bulk fetching (small demo)."""
    print("\n" + "="*70)
    print("🚀 EXAMPLE 4: Bulk Fetching Demo")
    print("="*70)
    
    # Set up dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')  # Just 1 week for demo
    
    # Get a small sample for actual fetching
    sample_stocks = ['0700.HK', '0005.HK', '0941.HK']  # Tencent, HSBC, China Mobile
    
    print(f"📅 Fetching {len(sample_stocks)} stocks from {start_date} to {end_date}")
    print(f"🎯 Sample stocks: {sample_stocks}")
    
    try:
        # Fetch the data
        stock_data = fetch_hk_stocks_bulk(
            tickers=sample_stocks,
            start_date=start_date,
            end_date=end_date,
            batch_size=2,
            delay_between_batches=1.0
        )
        
        print(f"\n✅ Successfully fetched {len(stock_data)} stocks")
        
        # Create summary
        if stock_data:
            summary = create_bulk_fetch_summary(stock_data)
            print("\n📊 Fetch Summary:")
            print(summary.to_string(index=False))
            
    except Exception as e:
        print(f"❌ Demo fetch failed: {e}")
        print("💡 This is normal - may be network/API issues")


def example_5_production_strategies():
    """Example 5: Production strategies for fetching all HK stocks."""
    print("\n" + "="*70)
    print("🏭 EXAMPLE 5: Production Strategies")
    print("="*70)
    
    print("🎯 STRATEGY 1: Conservative Approach")
    print("   • Start with major stocks (~50 stocks)")
    print("   • Batch size: 10-15 stocks")
    print("   • Delay: 3-5 seconds between batches")
    print("   • Total time: ~15-30 minutes")
    print("   • Success rate: 95%+")
    
    print("\n⚡ STRATEGY 2: Balanced Approach") 
    print("   • Fetch 100-200 stocks")
    print("   • Batch size: 20 stocks")
    print("   • Delay: 2-3 seconds between batches")
    print("   • Total time: ~30-60 minutes")
    print("   • Success rate: 90%+")
    
    print("\n🚀 STRATEGY 3: Comprehensive Approach")
    print("   • Fetch 500+ stocks")
    print("   • Batch size: 25-30 stocks")
    print("   • Delay: 2-3 seconds between batches")
    print("   • Total time: 2-4 hours")
    print("   • Success rate: 85%+")
    print("   • Recommendation: Run overnight")
    
    print("\n⚠️  IMPORTANT CONSIDERATIONS:")
    print("   • Yahoo Finance rate limits")
    print("   • Network stability")
    print("   • Cache utilization")
    print("   • Error recovery strategies")


def example_6_code_templates():
    """Example 6: Ready-to-use code templates."""
    print("\n" + "="*70)
    print("📝 EXAMPLE 6: Ready-to-Use Code Templates")
    print("="*70)
    
    templates = {
        "Fetch Top 50 HK Stocks": '''
# Fetch top 50 HK stocks
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
''',
        
        "Fetch by Sector": '''
# Fetch tech stocks
tech_data = fetch_hk_tech_stocks(
    start_date='2023-01-01',
    end_date='2024-01-01',
    batch_size=5,
    delay_between_batches=3.0
)

# Fetch finance stocks  
finance_data = fetch_hk_finance_stocks(
    start_date='2023-01-01',
    end_date='2024-01-01'
)
''',
        
        "Custom Bulk Fetch": '''
# Custom stock list
my_stocks = ['0700.HK', '0005.HK', '9988.HK', '1299.HK']

stock_data = fetch_hk_stocks_bulk(
    tickers=my_stocks,
    start_date='2023-01-01',
    end_date='2024-01-01',
    batch_size=2,
    delay_between_batches=2.0,
    force_refresh=False,
    skip_failed=True
)
''',
        
        "Save and Analyze": '''
# Save bulk data
save_bulk_data(
    stock_data=stock_data,
    base_dir="my_hk_stocks",
    create_summary=True
)

# Create analysis summary
summary_df = create_bulk_fetch_summary(stock_data)
print(summary_df.describe())
'''
    }
    
    for template_name, code in templates.items():
        print(f"\n📋 {template_name}:")
        print("   " + "-" * 50)
        print(code.strip())


def main():
    """Run all examples."""
    print("🚀 Hong Kong Stock Bulk Fetching Examples")
    print("=" * 70)
    print("This script demonstrates various approaches to fetch HK stock data")
    print("in bulk, from small targeted lists to comprehensive market coverage.")
    
    # Run examples
    example_1_sector_based_fetching()
    example_2_top_stocks()
    example_3_comprehensive_list()
    example_4_bulk_fetching_demo()
    example_5_production_strategies()
    example_6_code_templates()
    
    print("\n" + "="*70)
    print("🎉 All examples completed!")
    print("💡 Choose the approach that best fits your needs")
    print("⚠️  Remember to respect API rate limits")
    print("📓 Use the Jupyter notebooks for interactive exploration")


if __name__ == "__main__":
    main() 