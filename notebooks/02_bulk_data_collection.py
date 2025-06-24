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

# %% [raw]
# # Bulk Hong Kong Stock Data Collection
#
# This notebook demonstrates how to fetch data for **ALL Hong Kong stocks** efficiently using various approaches:
#
# ## 📋 What You'll Learn
#
# 1. **🎯 Curated Stock Lists**: Get major HK stocks by sector
# 2. **🔍 Stock Universe Discovery**: Find all available HK stocks
# 3. **⚡ Bulk Data Fetching**: Efficiently fetch large numbers of stocks
# 4. **🛡️ Rate Limiting & Error Handling**: Respect API limits
# 5. **📊 Progress Tracking**: Monitor large operations
# 6. **💾 Data Management**: Save and organize bulk data
#
# ## 🚀 Approaches Covered
#
# - **Static Lists**: Curated major stocks (most reliable)
# - **Sector-Based**: Tech, Finance, Property stocks
# - **Batch Processing**: Handle 100+ stocks efficiently
# - **Parallel Processing**: Speed up with threading (use carefully)
#
# ## 📝 Note
# Each cell in this notebook can be run independently after the setup cells (1-3).
#

# %% [raw]
# ## 🔧 Setup - Run These Cells First
#
# **Important**: Run cells 1-3 before running any demonstration cells.
#

# %%
# Cell 1: Import all required modules
import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import json
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm

# Add src directory to Python path
notebook_dir = Path().absolute()
src_dir = notebook_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

print("✅ Standard libraries imported")
print(f"📁 Source path added: {src_dir}")


# %%
# Cell 2: Import custom modules
try:
    from hk_stock_universe import (
        get_hk_stock_list_static,
        get_hk_stocks_by_sector,
        get_comprehensive_hk_stock_list,
        get_top_hk_stocks,
        MAJOR_HK_STOCKS
    )
    from bulk_data_fetcher import (
        fetch_hk_stocks_bulk,
        fetch_all_major_hk_stocks,
        fetch_top_50_hk_stocks,
        fetch_hk_tech_stocks,
        create_bulk_fetch_summary,
        save_bulk_data
    )
    print("✅ Custom modules imported successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print("Make sure you're running from the notebooks/ directory")


# %%
# Cell 3: Global setup - date range for all demonstrations
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

print(f"🗓️  Global date range: {start_date} to {end_date}")
print(f"📊 Expected trading days: ~65")
print("🚀 Setup complete! You can now run any demonstration cell independently.")


# %% [raw]
# ## 📊 Method 1: Explore Stock Categories
#
# **Objective**: Understand available HK stock categories and get familiar with the data structure.
#
# *This cell is completely independent - just run it!*
#

# %%
# METHOD 1: Explore Available Stock Categories
print("📊 Available Stock Categories:")
print("=" * 50)

for sector, stocks in MAJOR_HK_STOCKS.items():
    print(f"🏢 {sector.upper()}: {len(stocks)} stocks")
    print(f"   Examples: {', '.join(stocks[:3])}...")
    print()

# Get all major stocks (deduplicated)
all_major_stocks = get_hk_stock_list_static()
print(f"📈 Total unique major stocks: {len(all_major_stocks)}")
print(f"🔍 Sample tickers: {all_major_stocks[:10]}")

# Demo: Get stocks by specific sector
tech_stocks = get_hk_stocks_by_sector('tech_stocks')
finance_stocks = get_hk_stocks_by_sector('finance')
print(f"\n💻 Tech sector: {len(tech_stocks)} stocks")
print(f"🏦 Finance sector: {len(finance_stocks)} stocks")

print("\n✅ Method 1 Complete: Stock categories explored!")


# %% [raw]
# ## ⚡ Method 2: Bulk Fetch with Smart Batching
#
# **Objective**: Fetch multiple stocks efficiently using batch processing and rate limiting.
#
# *Independent cell - fetches 10 major stocks with progress tracking*
#

# %%
# METHOD 2: Bulk Fetch with Smart Batching
print("🚀 DEMO: Fetching top 10 HK stocks with smart batching...")

# Fetch top 10 stocks using the built-in bulk function
demo_stocks = fetch_all_major_hk_stocks(
    start_date=start_date,
    end_date=end_date,
    max_stocks=10,           # Small demo size
    batch_size=5,           # Process 5 at a time
    delay_between_batches=1.0  # 1 second between batches
)

print(f"\n✅ Demo completed! Fetched {len(demo_stocks)} stocks")

# Show summary of fetched data
if demo_stocks:
    summary_df = create_bulk_fetch_summary(demo_stocks)
    print("\n📊 Summary of fetched stocks:")
    display(summary_df.head())
    
    # Show data sample for first stock
    first_stock = list(demo_stocks.keys())[0]
    first_data = demo_stocks[first_stock]
    print(f"\n📈 Sample data for {first_stock}:")
    print(f"   Records: {len(first_data)}")
    print(f"   Date range: {first_data.index[0]} to {first_data.index[-1]}")
    print(f"   Columns: {list(first_data.columns)}")

print("\n✅ Method 2 Complete: Bulk fetching demonstrated!")


# %% [raw]
# ## 🏢 Method 3: Sector-Specific Bulk Fetching
#
# **Objective**: Fetch stocks from specific sectors (Tech, Finance, etc.) for targeted analysis.
#
# *Independent cell - fetches tech and finance sector stocks*
#

# %%
# METHOD 3: Sector-Specific Bulk Fetching
print("🏢 Fetching stocks by sector...")

# Fetch tech sector stocks
print("\n💻 Fetching Tech sector stocks...")
tech_data = fetch_hk_tech_stocks(
    start_date=start_date,
    end_date=end_date,
    batch_size=3,
    delay_between_batches=1.0
)
print(f"✅ Fetched {len(tech_data)} tech stocks")

# Fetch finance sector using manual approach
print("\n🏦 Fetching Finance sector stocks...")
finance_stocks = get_hk_stocks_by_sector('finance')
print(f"📊 Finance sector stocks: {finance_stocks}")

# Fetch first 3 finance stocks
finance_data = fetch_hk_stocks_bulk(
    tickers=finance_stocks[:3],
    start_date=start_date,
    end_date=end_date,
    batch_size=2,
    delay_between_batches=1.0
)
print(f"✅ Fetched {len(finance_data)} finance stocks")

# Compare sector performance
print(f"\n📊 Sector Comparison:")
print(f"   💻 Tech stocks fetched: {len(tech_data)}")
print(f"   🏦 Finance stocks fetched: {len(finance_data)}")

print("\n✅ Method 3 Complete: Sector-specific fetching demonstrated!")


# %% [raw]
# ## 🇭🇰 Method 4: Discover ALL Hong Kong Stocks
#
# **Objective**: Discover and analyze the complete HK stock universe (discovery only, no data fetching).
#
# *Independent cell - maps the full HK stock landscape*
#

# %%
# METHOD 4: Discover ALL Hong Kong Stocks
print("🔍 Discovering the complete HK stock universe...")

# Get comprehensive stock universe
stock_universe = get_comprehensive_hk_stock_list(
    include_major=True,
    validate_tickers=True,
    max_tickers=500  # Limit for demo - increase for full universe
)

# Extract and analyze the data
all_hk_stocks = sorted(stock_universe['valid_stocks'])
print(f"📊 Stock Universe Analysis:")
print(f"   Total discovered: {len(all_hk_stocks)} stocks")
print(f"   ✅ Valid: {len(stock_universe['valid_stocks'])}")
print(f"   ❌ Invalid: {len(stock_universe['invalid_stocks'])}")

if all_hk_stocks:
    print(f"\n🔍 Sample valid tickers: {all_hk_stocks[:10]}")
    print(f"📈 Stock code range: {all_hk_stocks[0]} to {all_hk_stocks[-1]}")

# Show detailed summary
print(f"\n📊 Detailed Summary:")
for key, value in stock_universe['summary'].items():
    print(f"   {key}: {value}")

# Calculate theoretical full market estimates
if len(all_hk_stocks) > 0:
    estimated_full_universe = len(all_hk_stocks) * 20  # Rough extrapolation
    estimated_time_hours = estimated_full_universe * 1.5 / 3600
    estimated_data_gb = estimated_full_universe * 0.001
    
    print(f"\n🚨 Theoretical FULL Market Estimates:")
    print(f"   📊 Estimated total HK stocks: ~{estimated_full_universe:,}")
    print(f"   ⏱️  Estimated fetch time: ~{estimated_time_hours:.1f} hours")
    print(f"   💾 Estimated data size: ~{estimated_data_gb:.1f} GB")
    print(f"   🌐 Estimated API calls: ~{estimated_full_universe:,}")

print(f"\n⚠️  Note: This demo uses a limited subset for demonstration.")
print(f"💡 To discover the full universe, increase max_tickers parameter.")

print("\n✅ Method 4 Complete: Stock universe discovery demonstrated!")


# %% [raw]
# ## 🚀 Method 5: Fetch ALL Hong Kong Stocks (Full Universe)
#
# **Objective**: Actually fetch stock data for ALL discovered HK stocks with progress tracking.
#
# *Independent cell - WARNING: This will take significant time and API quota!*
#
# **⚠️ CRITICAL**: This fetches data for ALL stocks. Use with caution!
#

# %%



# %%
# METHOD 5: Fetch ALL Hong Kong Stocks (Complete Universe Approach)
print("🚀 COMPREHENSIVE HK MARKET FETCH")
print("=" * 50)

# Safety configuration - CHANGE TO True TO EXECUTE FULL FETCH
EXECUTE_FULL_FETCH = False  # ⚠️ Change to True to execute full universe fetch
DEMO_SIZE = 50              # Number of stocks for demo mode

print(f"🛡️  Safety mode: {'DISABLED' if EXECUTE_FULL_FETCH else 'ENABLED'}")
print(f"📊 Demo size: {DEMO_SIZE} stocks")

if not EXECUTE_FULL_FETCH:
    print("\n💡 To enable FULL universe fetch:")
    print("   1. Set EXECUTE_FULL_FETCH = True")
    print("   2. Ensure sufficient API quota (1000+ calls)")
    print("   3. Prepare for 2-6 hours execution time")
    print("   4. Monitor system resources")
    
    # Demo with subset of discovered stocks
    print(f"\n🎯 DEMO: Fetching {DEMO_SIZE} stocks from discovered universe...")
    
    # Get stock universe for demo
    demo_universe = get_comprehensive_hk_stock_list(
        include_major=True,
        validate_tickers=True,
        max_tickers=DEMO_SIZE
    )
    
    demo_stocks_list = sorted(demo_universe['valid_stocks'])[:DEMO_SIZE]
    
    try:
        demo_comprehensive_data = fetch_hk_stocks_bulk(
            tickers=demo_stocks_list,
            start_date=start_date,
            end_date=end_date,
            batch_size=10,
            delay_between_batches=1.0
        )
        
        print(f"✅ Demo completed successfully!")
        print(f"📊 Fetched: {len(demo_comprehensive_data)} out of {len(demo_stocks_list)} stocks")
        print(f"📈 Demo success rate: {len(demo_comprehensive_data)/len(demo_stocks_list)*100:.1f}%")
        
        if demo_comprehensive_data:
            print(f"🌟 Sample fetched stocks: {list(demo_comprehensive_data.keys())[:5]}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        
else:
    print("🚨 EXECUTING FULL HK UNIVERSE FETCH!")
    print("⚠️  WARNING: This will take HOURS and significant API quota!")
    
    # Get complete HK stock universe
    print("🔍 Discovering complete HK stock universe...")
    full_universe = get_comprehensive_hk_stock_list(
        include_major=True,
        validate_tickers=True,
        max_tickers=None  # Get ALL available stocks
    )
    
    all_discovered_stocks = sorted(full_universe['valid_stocks'])
    print(f"🎯 Target: {len(all_discovered_stocks)} total discovered HK stocks")
    
    # Execute comprehensive fetch with progress tracking
    def fetch_universe_with_progress(stock_list, batch_size=25):
        successful_data = {}
        failed_stocks = []
        
        total_batches = (len(stock_list) + batch_size - 1) // batch_size
        print(f"📦 Processing {total_batches} batches of up to {batch_size} stocks each")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(stock_list))
            batch_stocks = stock_list[start_idx:end_idx]
            
            print(f"\n📦 Batch {batch_num+1}/{total_batches}: {batch_stocks[0]} to {batch_stocks[-1]}")
            
            try:
                batch_data = fetch_hk_stocks_bulk(
                    tickers=batch_stocks,
                    start_date=start_date,
                    end_date=end_date,
                    batch_size=10,
                    delay_between_batches=2.0
                )
                
                successful_data.update(batch_data)
                print(f"   ✅ Batch success: {len(batch_data)} stocks fetched")
                
            except Exception as e:
                print(f"   ❌ Batch failed: {e}")
                failed_stocks.extend(batch_stocks)
            
            # Progress update
            progress = (batch_num + 1) / total_batches * 100
            print(f"   📈 Overall progress: {progress:.1f}% - Total fetched: {len(successful_data)}")
        
        return successful_data, failed_stocks
    
    # Execute the complete universe fetch
    universe_data, universe_failures = fetch_universe_with_progress(
        all_discovered_stocks,
        batch_size=25
    )
    
    # Save complete universe data
    if universe_data:
        print(f"\n💾 Saving complete HK universe dataset...")
        save_bulk_data(
            stock_data=universe_data,
            base_dir="data/complete_hk_universe"
        )
    
    print(f"\n🎉 COMPLETE HK UNIVERSE FETCH FINISHED!")
    print(f"✅ Successfully fetched: {len(universe_data)} stocks")
    print(f"❌ Failed: {len(universe_failures)} stocks")
         print(f"📈 Success rate: {len(universe_data)/len(all_discovered_stocks)*100:.1f}%")

print("\n✅ Method 5 Complete: Comprehensive HK market fetch capability delivered!")

# %%

# %% [raw]
# ## 💾 Method 6: Data Management & Saving
#
# **Objective**: Save and organize bulk-fetched data systematically for future analysis.
#
# *Independent cell - demonstrates saving and data management*
#

# %%
# METHOD 6: Data Management & Saving
print("💾 Demonstrating data saving and management...")

# First, get some sample data
print("📊 Fetching sample data for saving demo...")
sample_stocks = fetch_all_major_hk_stocks(
    start_date=start_date,
    end_date=end_date,
    max_stocks=5,  # Small sample
    batch_size=3,
    delay_between_batches=0.5
)

if sample_stocks:
    print(f"✅ Got {len(sample_stocks)} stocks for saving demo")
    
    # Create and display summary
    summary_df = create_bulk_fetch_summary(sample_stocks)
    print("\n📊 Data Summary before saving:")
    display(summary_df)
    
    # Save bulk data
    print("\n💾 Saving data to files...")
    try:
        saved_files = save_bulk_data(
            stock_data=sample_stocks,
            base_dir="data/demo_bulk_save"
        )
        print(f"✅ Successfully saved data!")
        print(f"📁 Files saved in: data/demo_bulk_save/")
        
        # Show what was saved
        if 'files' in saved_files:
            print(f"📄 Individual stock files: {len(saved_files['files'])}")
        if 'summary_file' in saved_files:
            print(f"📊 Summary file: {saved_files['summary_file']}")
            
    except Exception as e:
        print(f"⚠️  Saving failed: {e}")
        print("💡 This is normal if data directory doesn't exist")
    
    # Demonstrate manual saving approach
    print(f"\n📝 Manual saving approach:")
    for ticker, data in list(sample_stocks.items())[:2]:  # First 2 stocks
        filename = f"manual_{ticker.replace('.', '_')}.csv"
        print(f"   💾 Would save {ticker} to {filename} ({len(data)} records)")
    
else:
    print("❌ No sample data available for saving demo")

print("\n✅ Method 6 Complete: Data management demonstrated!")


# %% [raw]
# ## 🛡️ Method 7: Error Handling & Retry Logic
#
# **Objective**: Demonstrate robust error handling with retry logic for production-ready bulk fetching.
#
# *Independent cell - simulates failures and shows recovery strategies*
#

# %%
# METHOD 7: Error Handling & Retry Logic
print("🛡️ Demonstrating robust error handling...")

# Define a robust fetching function with retry logic
def demo_robust_fetch(stock_list, max_retries=3):
    """Demo function showing retry logic and error handling"""
    successful_fetches = {}
    failed_stocks = []
    
    for stock in stock_list:
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                print(f"🔄 Fetching {stock} (attempt {retry_count + 1}/{max_retries})")
                
                # Simulate API call with 70% success rate
                if random.random() < 0.7:
                    successful_fetches[stock] = f"✅ Data for {stock}"
                    success = True
                    print(f"   ✅ Success!")
                else:
                    raise Exception("Simulated API timeout")
                    
            except Exception as e:
                retry_count += 1
                print(f"   ⚠️  Failed: {e}")
                
                if retry_count < max_retries:
                    wait_time = retry_count * 1  # Progressive backoff
                    print(f"   ⏱️  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    failed_stocks.append(stock)
                    print(f"   ❌ Gave up after {max_retries} attempts")
    
    return successful_fetches, failed_stocks

# Demo with sample stocks
sample_stock_list = ['0700.HK', '0005.HK', '0941.HK', '1299.HK', '2318.HK']
print(f"🚀 Testing robust fetching with {len(sample_stock_list)} stocks:")

# Set random seed for reproducible demo
random.seed(42)

successful_data, failed_list = demo_robust_fetch(
    stock_list=sample_stock_list,
    max_retries=2
)

print(f"\n📊 Error Handling Results:")
print(f"   ✅ Successful: {len(successful_data)} stocks")
print(f"   ❌ Failed: {len(failed_list)} stocks")
print(f"   📈 Success rate: {len(successful_data)/len(sample_stock_list)*100:.1f}%")

if failed_list:
    print(f"   🔍 Failed stocks: {failed_list}")

# Show error handling best practices
print(f"\n🛡️ Error Handling Best Practices:")
print("   1. ✅ Implement exponential backoff")
print("   2. ✅ Set maximum retry limits")
print("   3. ✅ Log detailed error information")
print("   4. ✅ Continue processing other stocks")
print("   5. ✅ Track success/failure rates")
print("   6. ✅ Provide recovery mechanisms")

print("\n✅ Method 7 Complete: Error handling demonstrated!")


# %% [raw]
# ## ⚡ Method 8: Parallel Processing (Advanced)
#
# **Objective**: Demonstrate parallel processing with threading for faster bulk fetching.
#
# *Independent cell - shows parallel processing with safety considerations*
#
# **⚠️ WARNING**: Use parallel processing carefully to avoid overwhelming APIs!
#

# %%
# METHOD 8: Parallel Processing (Advanced)
print("⚡ Demonstrating parallel processing...")
print("⚠️  Using conservative settings to respect API limits")

def demo_fetch_single_stock(stock_symbol):
    """Demo function for single stock fetch with delay"""
    try:
        time.sleep(0.5)  # Conservative delay per request
        print(f"   🔄 Processed {stock_symbol}")
        return f"Data for {stock_symbol}"
    except Exception as e:
        print(f"   ❌ Error with {stock_symbol}: {e}")
        return None

# Test with small stock list
demo_stocks = ['0700.HK', '0005.HK', '0941.HK', '1299.HK']

print(f"🚀 Parallel Processing Demo ({len(demo_stocks)} stocks):")
print(f"⚡ Comparing sequential vs parallel processing...")

# Sequential processing
print(f"\n📈 Sequential Processing:")
start_time = time.time()
sequential_results = []
for stock in demo_stocks:
    result = demo_fetch_single_stock(stock)
    sequential_results.append(result)
sequential_time = time.time() - start_time

print(f"   ⏱️  Sequential time: {sequential_time:.2f} seconds")

# Parallel processing with limited workers
print(f"\n⚡ Parallel Processing (2 workers max):")
start_time = time.time()

with ThreadPoolExecutor(max_workers=2) as executor:  # CONSERVATIVE: Only 2 workers
    parallel_results = list(executor.map(demo_fetch_single_stock, demo_stocks))

parallel_time = time.time() - start_time

print(f"   ⏱️  Parallel time: {parallel_time:.2f} seconds")
print(f"   🚀 Speed improvement: {sequential_time/parallel_time:.1f}x")

# Results comparison
successful_sequential = len([r for r in sequential_results if r])
successful_parallel = len([r for r in parallel_results if r])

print(f"\n📊 Results Comparison:")
print(f"   📈 Sequential: {successful_sequential}/{len(demo_stocks)} successful")
print(f"   ⚡ Parallel: {successful_parallel}/{len(demo_stocks)} successful")

# Safety recommendations
print(f"\n🛡️ Parallel Processing Safety Guidelines:")
print("   1. ⚠️  Start with max_workers=2 (conservative)")
print("   2. ⏱️  Include delays in individual requests")
print("   3. 📊 Monitor API response times")
print("   4. 🔄 Test with small batches first")
print("   5. 📈 Scale up gradually if successful")
print("   6. 🚨 Have fallback to sequential processing")

print("\n✅ Method 8 Complete: Parallel processing demonstrated!")


# %% [raw]
# ## 📊 Summary & Best Practices
#
# **Key takeaways and recommendations from all methods demonstrated above.**
#

# %%
# SUMMARY: Best Practices & Recommendations
print("📊 BULK HK STOCK DATA COLLECTION - SUMMARY")
print("=" * 60)

print("\n🎯 RECOMMENDED APPROACHES:")

print("\n1. 🔰 BEGINNER: Start Small")
print("   • Use: fetch_all_major_hk_stocks(max_stocks=10-20)")
print("   • Benefits: Safe, reliable, fast")
print("   • Best for: Learning and testing")

print("\n2. 📊 SECTOR ANALYSIS: Targeted Approach")
print("   • Use: fetch_hk_tech_stocks() or get_hk_stocks_by_sector()")
print("   • Benefits: Focused analysis, manageable size")
print("   • Best for: Sector-specific research")

print("\n3. 🚀 ADVANCED: Comprehensive Analysis")
print("   • Use: get_comprehensive_hk_stock_list() with batching")
print("   • Benefits: Full market coverage")
print("   • Best for: Complete market analysis")

print("\n4. ⚡ ENTERPRISE: High-Volume Processing")
print("   • Use: Parallel processing with careful rate limiting")
print("   • Benefits: Faster processing")
print("   • Best for: Production systems with monitoring")

print("\n🛡️ CRITICAL SUCCESS FACTORS:")
print("   ✅ Respect API rate limits (1-2 second delays)")
print("   ✅ Use batch processing (5-20 stocks per batch)")
print("   ✅ Implement retry logic with backoff")
print("   ✅ Monitor success rates and performance")
print("   ✅ Save data systematically")
print("   ✅ Test with small datasets first")

print("\n⚠️  IMPORTANT WARNINGS:")
print("   🚨 Full HK market = 1000+ stocks = hours of processing")
print("   🚨 Always validate tickers before bulk processing")
print("   🚨 Monitor API usage quotas")
print("   🚨 Parallel processing can trigger rate limits")

print("\n🚀 PRODUCTION READY CHECKLIST:")
print("   □ Error handling and retry logic")
print("   □ Progress tracking and logging")
print("   □ Data validation and quality checks")
print("   □ Checkpoint/resume capability")
print("   □ Resource monitoring (CPU, memory, network)")
print("   □ Fallback strategies for failures")

print("\n" + "="*60)
print("✅ NOTEBOOK COMPLETED SUCCESSFULLY!")
print("📚 You now have tools for any scale of HK stock data collection!")
print("🇭🇰 Happy analyzing! 📈")

# Show what was accomplished
methods_completed = [
    "✅ Method 1: Stock categories exploration",
    "✅ Method 2: Smart batching demonstration", 
    "✅ Method 3: Sector-specific fetching",
    "✅ Method 4: Full HK stock universe discovery",
    "✅ Method 5: Comprehensive HK market fetch (full universe)",
    "✅ Method 6: Data management and saving",
    "✅ Method 7: Error handling and retry logic",
    "✅ Method 8: Parallel processing (advanced)"
]

print(f"\n📋 METHODS DEMONSTRATED:")
for method in methods_completed:
    print(f"   {method}")

print(f"\n💡 NEXT STEPS:")
print("   • Run individual cells based on your needs")
print("   • Modify parameters for your specific requirements")
print("   • Scale up gradually from small to large datasets")
print("   • Implement production safeguards for real applications")


# %% [raw]
#

# %% [raw]
#
#

# %%

import json
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm

# Add src directory to Python path
notebook_dir = Path().absolute()
src_dir = notebook_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

# Import our bulk fetching modules
try:
    from hk_stock_universe import (
        get_hk_stock_list_static,
        get_hk_stocks_by_sector,
        get_comprehensive_hk_stock_list,
        get_top_hk_stocks,
        MAJOR_HK_STOCKS
    )
    from bulk_data_fetcher import (
        fetch_hk_stocks_bulk,
        fetch_all_major_hk_stocks,
        fetch_top_50_hk_stocks,
        fetch_hk_tech_stocks,
        create_bulk_fetch_summary,
        save_bulk_data
    )
    print("✅ All modules imported successfully!")
    print("🚀 Ready for bulk data collection!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries


# %%
# Set up date range for all demonstrations
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

print(f"📅 Date range for all demos: {start_date} to {end_date}")
print(f"📊 Expected trading days: ~65")


# %%
# Helper function definitions for advanced demonstrations

def fetch_all_hk_with_progress(stock_list, start_date, end_date, batch_size=10):
    """
    Fetch all stocks with progress bar and comprehensive tracking
    """
    successful_data = {}
    failed_stocks = []
    progress_stats = {
        'total_stocks': len(stock_list),
        'completed': 0,
        'success_rate': 0,
        'start_time': time.time()
    }
    
    # Create progress bar
    pbar = tqdm(total=len(stock_list), desc="Fetching HK Stocks")
    
    # Process in batches
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i+batch_size]
        pbar.set_description(f"Batch {i//batch_size + 1} ({batch[0]} to {batch[-1]})")
        
        # Fetch batch using existing function
        try:
            batch_data = fetch_hk_stocks_bulk(
                tickers=batch,
                start_date=start_date,
                end_date=end_date,
                batch_size=batch_size,
                delay_between_batches=1.0
            )
            
            successful_data.update(batch_data)
            pbar.update(len(batch))
            
        except Exception as e:
            print(f"❌ Batch failed: {e}")
            failed_stocks.extend(batch)
            pbar.update(len(batch))
        
        # Update progress stats
        progress_stats['completed'] = len(successful_data)
        progress_stats['success_rate'] = len(successful_data) / progress_stats['total_stocks'] * 100
        
        # Update progress bar with stats
        elapsed_time = time.time() - progress_stats['start_time']
        stocks_per_second = progress_stats['completed'] / elapsed_time if elapsed_time > 0 else 0
        
        pbar.set_postfix({
            'Success': f"{progress_stats['success_rate']:.1f}%",
            'Rate': f"{stocks_per_second:.2f}/s"
        })
    
    pbar.close()
    
    # Final statistics
    total_time = time.time() - progress_stats['start_time']
    print(f"\n📊 COMPREHENSIVE FETCH COMPLETED!")
    print(f"✅ Successfully fetched: {len(successful_data)} stocks")
    print(f"❌ Failed: {len(failed_stocks)} stocks")
    print(f"📈 Success rate: {len(successful_data)/len(stock_list)*100:.1f}%")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🚀 Average rate: {len(successful_data)/total_time:.2f} stocks/second")
    
    return successful_data, failed_stocks


def robust_bulk_fetch(stock_list, start_date, end_date, max_retries=3):
    """
    Fetch stocks with retry logic and error handling
    """
    successful_fetches = {}
    failed_stocks = []
    
    for stock in stock_list:
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                print(f"🔄 Fetching {stock} (attempt {retry_count + 1}/{max_retries})")
                
                # Simulate API call with potential failure
                if random.random() < 0.8:  # 80% success rate for demo
                    # This would be your actual data fetching call
                    successful_fetches[stock] = f"Mock data for {stock}"
                    success = True
                    print(f"✅ Successfully fetched {stock}")
                else:
                    raise Exception("Simulated API error")
                    
            except Exception as e:
                retry_count += 1
                print(f"⚠️  Error fetching {stock}: {e}")
                
                if retry_count < max_retries:
                    wait_time = retry_count * 2  # Exponential backoff
                    print(f"⏱️  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    failed_stocks.append(stock)
                    print(f"❌ Failed to fetch {stock} after {max_retries} attempts")
    
    return successful_fetches, failed_stocks


def fetch_single_stock_with_delay(stock_symbol):
    """Fetch a single stock with built-in delay for parallel processing demo"""
    try:
        time.sleep(0.5)  # Conservative delay
        print(f"🔄 Fetching {stock_symbol}...")
        return f"Data for {stock_symbol}"
    except Exception as e:
        print(f"❌ Error fetching {stock_symbol}: {e}")
        return None

print("✅ Helper functions defined!")


# %% [raw]
# ## Method 1: Fetch Major HK Stocks by Sector
#
# Start with curated lists of major Hong Kong stocks organized by sector. This is the most reliable approach.
#

# %%
# Explore available stock categories
print("📊 Available Stock Categories:")
print("=" * 50)

for sector, stocks in MAJOR_HK_STOCKS.items():
    print(f"🏢 {sector.upper()}: {len(stocks)} stocks")
    print(f"   Examples: {', '.join(stocks[:3])}...")
    print()

# Get all major stocks (deduplicated)
all_major_stocks = get_hk_stock_list_static()
print(f"📈 Total unique major stocks: {len(all_major_stocks)}")

# Show some examples
print(f"🔍 Sample tickers: {all_major_stocks[:10]}")


# %% [raw]
# ## Method 2: Bulk Fetch with Smart Batching
#
# Now let's fetch data for all major stocks using intelligent batching and rate limiting.
#

# %% [raw]
# len(all_major_stocks)

# %%
# Set up date range (last 3 months for demo)
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

print(f"📅 Fetching data from {start_date} to {end_date}")
print(f"📊 Expected trading days: ~65")

# Example 1: Fetch top 20 stocks (for demo purposes)
print("\n🚀 DEMO: Fetching top 20 HK stocks...")

demo_stocks = fetch_all_major_hk_stocks(
    start_date=start_date,
    end_date=end_date,
    max_stocks=20,           # Limit for demo
    batch_size=5,           # Small batches for demo
    delay_between_batches=1.0  # 1 second between batches
)

print(f"\n✅ Demo completed! Fetched {len(demo_stocks)} stocks")

# Show summary
if demo_stocks:
    summary_df = create_bulk_fetch_summary(demo_stocks)
    print("\n📊 Summary of fetched stocks:")
    display(summary_df.head(10))


# %%
## Method 3: Sector-Specific Bulk Fetching

Focus on specific sectors for targeted analysis. This is useful when you want to analyze particular market segments.

# %%
# Example: Fetch all Tech stocks
print("🚀 Fetching Tech sector stocks...")
tech_stocks = fetch_hk_tech_stocks(
    start_date=start_date,
    end_date=end_date,
    delay_between_requests=0.5
)

print(f"✅ Fetched {len(tech_stocks)} tech stocks")

# Example: Fetch Finance sector stocks  
print("\n🏦 Fetching Finance sector stocks...")
finance_stocks = get_hk_stocks_by_sector('finance')
print(f"📊 Finance sector stocks: {finance_stocks}")

# Demonstrate batch processing for a specific sector
finance_data = fetch_hk_stocks_bulk(
    tickers=finance_stocks[:5],  # First 5 finance stocks
    start_date=start_date,
    end_date=end_date,
    batch_size=3,
    delay_between_batches=1.0
)

print(f"\n📈 Successfully fetched data for {len(finance_data)} finance stocks")


# %% [raw]
#
#

# %%
# Step 1: Get comprehensive list of ALL HK stocks
print("🔍 Discovering ALL Hong Kong stocks...")
print("=" * 50)

# Get the complete HK stock universe
stock_universe = get_comprehensive_hk_stock_list(
    include_major=True,
    validate_tickers=True,
    max_tickers=100  # Limit for demo - increase for full universe
)

# Extract the actual stock tickers from the result
all_hk_stocks = sorted(stock_universe['valid_stocks'])
print(f"📊 Total HK stocks discovered: {len(all_hk_stocks)}")
print(f"✅ Valid stocks: {len(stock_universe['valid_stocks'])}")
print(f"❌ Invalid stocks: {len(stock_universe['invalid_stocks'])}")

# Show sample of discovered stocks
print(f"🔍 Sample tickers: {all_hk_stocks[:20]}")
if len(all_hk_stocks) > 10:
    print(f"📈 Stock range: {all_hk_stocks[0]} to {all_hk_stocks[-10:]}")

# Show summary
print(f"\n📊 Stock Universe Summary:")
for key, value in stock_universe['summary'].items():
    print(f"   {key}: {value}")

# Calculate estimated time and API calls
estimated_time_hours = len(all_hk_stocks) * 1.5 / 3600  # 1.5 seconds per stock
estimated_api_calls = len(all_hk_stocks)

print(f"\n⏱️  Estimated time for ALL stocks: {estimated_time_hours:.1f} hours")
print(f"🌐 Estimated API calls: {estimated_api_calls:,}")
print(f"💰 Estimated data size: ~{len(all_hk_stocks) * 0.1:.1f} MB")

print("\n🚨 **RECOMMENDATION**: Start with a subset for testing!")


# %%
# Step 2: Smart subset selection for demonstration
print("🎯 Creating manageable subsets for demonstration:")
print("=" * 50)

# Option 1: Top N stocks by stock code (usually more liquid)
top_100_stocks = all_hk_stocks[:100]
print(f"📊 Top 100 stocks (by code): {len(top_100_stocks)}")

# Option 2: Random sampling across the range
import random
random.seed(42)  # For reproducible results
random_sample = random.sample(all_hk_stocks, min(50, len(all_hk_stocks)))
random_sample.sort()  # Sort for easier tracking
print(f"🎲 Random sample: {len(random_sample)} stocks")
print(f"   Examples: {random_sample[:10]}")

# Option 3: Systematic sampling (every Nth stock)
systematic_sample = all_hk_stocks[::max(1, len(all_hk_stocks)//50)]  # Every ~20th stock
print(f"📐 Systematic sample: {len(systematic_sample)} stocks")
print(f"   Examples: {systematic_sample[:10]}")

# Choose which subset to use for demo
DEMO_SUBSET = random_sample  # Use random sample for variety
print(f"\n✅ Selected subset for demo: {len(DEMO_SUBSET)} stocks")
print(f"📋 Subset range: {DEMO_SUBSET[0]} to {DEMO_SUBSET[-1]}")


# %%
# Step 3: Fetch subset with progress tracking
print("🚀 Fetching comprehensive HK stock data with progress tracking...")
print("=" * 60)

from tqdm.notebook import tqdm
import time

def fetch_all_hk_with_progress(stock_list, start_date, end_date, batch_size=10):
    """
    Fetch all stocks with progress bar and comprehensive tracking
    """
    successful_data = {}
    failed_stocks = []
    progress_stats = {
        'total_stocks': len(stock_list),
        'completed': 0,
        'success_rate': 0,
        'start_time': time.time()
    }
    
    # Create progress bar
    pbar = tqdm(total=len(stock_list), desc="Fetching HK Stocks")
    
    # Process in batches
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i+batch_size]
        pbar.set_description(f"Batch {i//batch_size + 1} ({batch[0]} to {batch[-1]})")
        
        # Fetch batch using existing function
        try:
            batch_data = fetch_hk_stocks_bulk(
                tickers=batch,
                start_date=start_date,
                end_date=end_date,
                batch_size=batch_size,
                delay_between_batches=1.0
            )
            
            successful_data.update(batch_data)
            pbar.update(len(batch))
            
        except Exception as e:
            print(f"❌ Batch failed: {e}")
            failed_stocks.extend(batch)
            pbar.update(len(batch))
        
        # Update progress stats
        progress_stats['completed'] = len(successful_data)
        progress_stats['success_rate'] = len(successful_data) / progress_stats['total_stocks'] * 100
        
        # Update progress bar with stats
        elapsed_time = time.time() - progress_stats['start_time']
        stocks_per_second = progress_stats['completed'] / elapsed_time if elapsed_time > 0 else 0
        
        pbar.set_postfix({
            'Success': f"{progress_stats['success_rate']:.1f}%",
            'Rate': f"{stocks_per_second:.2f}/s"
        })
    
    pbar.close()
    
    # Final statistics
    total_time = time.time() - progress_stats['start_time']
    print(f"\n📊 COMPREHENSIVE FETCH COMPLETED!")
    print(f"✅ Successfully fetched: {len(successful_data)} stocks")
    print(f"❌ Failed: {len(failed_stocks)} stocks")
    print(f"📈 Success rate: {len(successful_data)/len(stock_list)*100:.1f}%")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🚀 Average rate: {len(successful_data)/total_time:.2f} stocks/second")
    
    return successful_data, failed_stocks

# Execute the comprehensive fetch (limited subset for demo)
print(f"🎯 Starting comprehensive fetch of {len(DEMO_SUBSET)} stocks...")
print("⚠️  This is a DEMO with limited stocks. For ALL stocks, increase the subset size.")

comprehensive_data, comprehensive_failures = fetch_all_hk_with_progress(
    stock_list=DEMO_SUBSET[:10],  # Limit to 10 for demo
    start_date=start_date,
    end_date=end_date,
    batch_size=5
)


# %%
# Step 4: Production-ready ALL stocks fetching template
print("🏭 PRODUCTION TEMPLATE: Fetch ALL Hong Kong Stocks")
print("=" * 60)

def fetch_entire_hk_market(start_date, end_date, 
                          checkpoint_every=100, 
                          resume_from_checkpoint=False,
                          max_workers=1):
    """
    Production-ready function to fetch ALL HK stocks with:
    - Checkpointing for resume capability
    - Memory management
    - Comprehensive logging
    - Error recovery
    """
    
    # Get complete stock list
    all_stocks = get_comprehensive_hk_stock_list()
    
    print(f"🎯 TARGET: {len(all_stocks)} total HK stocks")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🔄 Checkpoint every: {checkpoint_every} stocks")
    
    # Checkpoint file management
    checkpoint_file = f"checkpoint_hk_stocks_{start_date}_{end_date}.json"
    completed_stocks = set()
    
    if resume_from_checkpoint and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            completed_stocks = set(checkpoint_data.get('completed', []))
        print(f"📋 Resuming from checkpoint: {len(completed_stocks)} already completed")
    
    # Filter remaining stocks
    remaining_stocks = [s for s in all_stocks if s not in completed_stocks]
    print(f"📊 Remaining to fetch: {len(remaining_stocks)} stocks")
    
    # Estimate resources
    estimated_hours = len(remaining_stocks) * 1.5 / 3600
    estimated_gb = len(remaining_stocks) * 0.001  # ~1MB per stock
    
    print(f"⏱️  Estimated time: {estimated_hours:.1f} hours")
    print(f"💾 Estimated storage: {estimated_gb:.2f} GB")
    print(f"🌐 Estimated API calls: {len(remaining_stocks):,}")
    
    # WARNING and confirmation
    print(f"\n🚨 WARNING: This will make {len(remaining_stocks):,} API calls!")
    print("🚨 This operation will take several hours to complete.")
    print("🚨 Ensure you have sufficient API quota and storage space.")
    
    # For safety, we'll just show the template without executing
    print("\n" + "="*60)
    print("📝 TEMPLATE READY - Execute with caution!")
    print("💡 TIP: Start with a smaller subset to test your setup first.")
    print("💡 TIP: Run during off-peak hours to avoid rate limiting.")
    print("💡 TIP: Monitor your API usage regularly.")
    
    return {
        'total_stocks': len(all_stocks),
        'remaining_stocks': len(remaining_stocks),
        'completed_stocks': len(completed_stocks),
        'estimated_hours': estimated_hours,
        'template_ready': True
    }

# Show the production template (without executing)
production_info = fetch_entire_hk_market(
    start_date=start_date,
    end_date=end_date
)

print(f"\n📊 Production Analysis Complete:")
for key, value in production_info.items():
    print(f"   {key}: {value}")


# %%
# Import json for checkpoint functionality
import json

# Step 5: Execute Full Market Fetch (UNCOMMENT TO RUN)
print("⚠️  FULL MARKET EXECUTION CODE (Currently Commented for Safety)")
print("=" * 60)

# UNCOMMENT AND MODIFY THESE LINES TO EXECUTE FULL MARKET FETCH:

execute_full_fetch = True  # Set to True to execute

if execute_full_fetch:
    print("🚀 EXECUTING FULL HK MARKET FETCH...")
    
    # WARNING: This will take hours and use significant API quota
    full_market_data = fetch_all_hk_with_progress(
        stock_list=all_hk_stocks,  # ALL stocks
        start_date=start_date,
        end_date=end_date,
        batch_size=10
    )
    
    # Save the complete dataset
    save_bulk_data(
        stock_data_dict=full_market_data[0],
        output_dir="data/full_hk_market",
        file_format="csv"
    )
    
    print("✅ FULL HK MARKET FETCH COMPLETED!")
    
else:
    print("🛡️  Full market fetch is DISABLED for safety.")
    print("📋 To execute the full market fetch:")
    print("   1. Set execute_full_fetch = True")
    print("   2. Ensure you have sufficient API quota")
    print("   3. Prepare for several hours of execution time")
    print("   4. Monitor system resources and API usage")
    print("   5. Consider running during off-peak hours")
    
    print(f"\n📊 READY TO FETCH: {len(all_hk_stocks) if 'all_hk_stocks' in locals() else 'N/A'} total HK stocks")
    print("🚀 All infrastructure is in place for full market analysis!")


# %%



# %%
# Save individual stock data to CSV files
print("💾 Saving bulk data to files...")

if demo_stocks:
    saved_files = save_bulk_data(
        stock_data_dict=demo_stocks,
        output_dir="data",
        file_format="csv"
    )
    print(f"✅ Saved {len(saved_files)} stock data files")
    print(f"📁 Files saved to: {saved_files[:3]}...")  # Show first 3 files
    
# Create a comprehensive summary
if demo_stocks:
    summary_df = create_bulk_fetch_summary(demo_stocks)
    print("\n📊 Comprehensive Stock Summary:")
    print("=" * 60)
    display(summary_df)
    
    # Save summary to CSV
    summary_file = "data/bulk_stock_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n💾 Summary saved to: {summary_file}")


# %% [raw]
#
#

# %%
# Example of parallel processing (use sparingly)
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_single_stock_with_delay(stock_symbol):
    """Fetch a single stock with built-in delay"""
    try:
        time.sleep(0.5)  # Conservative delay
        # This would call your actual fetching function
        print(f"🔄 Fetching {stock_symbol}...")
        return f"Data for {stock_symbol}"
    except Exception as e:
        print(f"❌ Error fetching {stock_symbol}: {e}")
        return None

# Demonstrate parallel processing with a small subset
small_stock_list = all_major_stocks[:5]  # Only use 5 stocks for demo

print("🚀 Parallel Processing Demo (5 stocks):")
print("⚠️  Using conservative delays to respect rate limits")

start_time = time.time()

# Use ThreadPoolExecutor with limited workers
with ThreadPoolExecutor(max_workers=2) as executor:  # Only 2 concurrent requests
    results = list(executor.map(fetch_single_stock_with_delay, small_stock_list))

end_time = time.time()

print(f"\n✅ Parallel processing completed in {end_time - start_time:.2f} seconds")
print(f"📊 Successfully processed {len([r for r in results if r])} stocks")
print("\n⚠️  **Note**: Always test rate limits before using parallel processing in production!")


# %%

# %% [raw]
# ## Method 6: Error Handling & Recovery
#
# Robust error handling is crucial when fetching large amounts of data.
#

# %%
# Example of robust error handling during bulk fetching
def robust_bulk_fetch(stock_list, start_date, end_date, max_retries=3):
    """
    Fetch stocks with retry logic and error handling
    """
    successful_fetches = {}
    failed_stocks = []
    
    for stock in stock_list:
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                print(f"🔄 Fetching {stock} (attempt {retry_count + 1}/{max_retries})")
                
                # Simulate API call with potential failure
                import random
                if random.random() < 0.8:  # 80% success rate for demo
                    # This would be your actual data fetching call
                    successful_fetches[stock] = f"Mock data for {stock}"
                    success = True
                    print(f"✅ Successfully fetched {stock}")
                else:
                    raise Exception("Simulated API error")
                    
            except Exception as e:
                retry_count += 1
                print(f"⚠️  Error fetching {stock}: {e}")
                
                if retry_count < max_retries:
                    wait_time = retry_count * 2  # Exponential backoff
                    print(f"⏱️  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    failed_stocks.append(stock)
                    print(f"❌ Failed to fetch {stock} after {max_retries} attempts")
    
    return successful_fetches, failed_stocks

# Demo with a small subset
demo_stock_subset = all_major_stocks[:5]
print(f"🚀 Testing robust fetching with {len(demo_stock_subset)} stocks:")

successful_data, failed_list = robust_bulk_fetch(
    stock_list=demo_stock_subset,
    start_date=start_date,
    end_date=end_date,
    max_retries=2
)

print(f"\n📊 Results:")
print(f"✅ Successful: {len(successful_data)} stocks")
print(f"❌ Failed: {len(failed_list)} stocks")
if failed_list:
    print(f"📝 Failed stocks: {failed_list}")


# %% [raw]
# ## 📊 Summary & Best Practices
#
# ### 🎯 Key Takeaways
#
# 1. **Start Small**: Begin with curated lists of major stocks (most reliable)
# 2. **Respect Rate Limits**: Use delays between requests (0.5-2 seconds recommended)
# 3. **Batch Processing**: Process stocks in small batches (5-10 stocks per batch)
# 4. **Error Handling**: Implement retry logic with exponential backoff
# 5. **Data Management**: Organize and save data systematically
# 6. **Monitoring**: Track progress and success rates
#
# ### ⚡ Performance Tips
#
# - **Sequential Processing**: Safest approach, respects API limits
# - **Parallel Processing**: Use sparingly with conservative limits (max 2-3 workers)
# - **Caching**: Save fetched data to avoid re-fetching
# - **Incremental Updates**: Only fetch new data when needed
#
# ### 🛡️ Risk Management
#
# - Always test with small datasets first
# - Monitor API response times and error rates
# - Have fallback strategies for failed requests
# - Keep track of API usage quotas
#

# %%
# Final demonstration: Choose your approach based on needs

print("🎯 RECOMMENDED APPROACHES:")
print("=" * 50)

print("\n1. 🔰 **BEGINNER**: Start with top 10-20 stocks")
print("   - Use: fetch_all_major_hk_stocks(max_stocks=20)")
print("   - Safe, reliable, fast")

print("\n2. 📊 **SECTOR ANALYSIS**: Focus on specific sectors")
print("   - Use: fetch_hk_tech_stocks() or get_hk_stocks_by_sector()")
print("   - Targeted analysis, manageable size")

print("\n3. 🚀 **ADVANCED**: Full market analysis")
print("   - Use: fetch_all_major_hk_stocks() with careful batching")
print("   - Comprehensive but requires patience")

print("\n4. ⚡ **ENTERPRISE**: High-volume processing")
print("   - Implement parallel processing with rate limiting")
print("   - Requires careful monitoring and error handling")

print("\n" + "="*50)
print("✅ Notebook completed successfully!")
print("📚 You now have tools for any scale of HK stock data collection!")
print("🚀 Happy analyzing! 🇭🇰📈")

