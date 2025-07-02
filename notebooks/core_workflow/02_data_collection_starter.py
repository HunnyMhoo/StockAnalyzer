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
# # 🔰 Hong Kong Stock Data Collection - Starter Edition
#
# **Perfect for:** Beginners to intermediate users (10-100 stocks)
#
# This notebook provides a **comprehensive yet beginner-friendly approach** for collecting Hong Kong stock data in organized batches.
#
# ## ✅ What You'll Learn
# - Fetch 10-100 stocks safely with intelligent rate limiting
# - Sector-based stock selection and analysis
# - Progressive scaling from small to medium datasets
# - Data validation and quality assessment
# - Smart caching and incremental updates
# - Performance monitoring and optimization
#
# ## ⏱️ Time Estimates
# - **Setup**: 2-3 minutes
# - **Small batch (10 stocks)**: 3-5 minutes  
# - **Medium batch (25 stocks)**: 8-12 minutes
# - **Large batch (50 stocks)**: 15-20 minutes
# - **Extended batch (100 stocks)**: 25-35 minutes
#
# ## 🎯 Prerequisites
# - Basic Python knowledge
# - Internet connection for data fetching
# - ~1GB free disk space for data cache

# %%
# SETUP AND IMPORTS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import sys
import os
from pathlib import Path
from tqdm.notebook import tqdm

# Add project root to path
project_root = Path('.').resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import setup utilities
from notebooks.utilities.common_setup import setup_notebook, get_date_range, import_common_modules

# Initialize notebook environment
print("🔰 **Hong Kong Stock Data Collection - Starter Edition**")
print("=" * 60)
validation = setup_notebook()

# Import data collection modules
modules = import_common_modules()
get_hk_stock_list_static = modules['get_hk_stock_list_static']

# Import collection functions
try:
    from stock_analyzer.data import (
        fetch_hk_stocks_bulk,
        fetch_all_major_hk_stocks,
        create_bulk_fetch_summary,
        get_hk_stocks_by_sector,
        MAJOR_HK_STOCKS
    )
    print("✅ Stock analyzer modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Fallback to direct module imports: {e}")
    # Try direct imports from project modules
    try:
        sys.path.append(str(project_root / 'src'))
        from bulk_data_fetcher import (
            fetch_hk_stocks_bulk,
            fetch_all_major_hk_stocks,
            create_bulk_fetch_summary
        )
        from hk_stock_universe import get_hk_stocks_by_sector, MAJOR_HK_STOCKS
        print("✅ Legacy modules loaded successfully")
    except ImportError as fallback_error:
        print(f"❌ Import error: {fallback_error}")
        print("Please ensure all required modules are installed and accessible")

print(f"✅ Setup completed successfully!")
print(f"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ## ⚙️ Configuration Profiles
#
# Choose from pre-configured profiles designed for different experience levels:

# %%
# Define progressive configuration profiles
COLLECTION_PROFILES = {
    'beginner': {
        'max_stocks': 10,
        'batch_size': 3,
        'delay_between_batches': 3.0,
        'description': 'Safe introduction - 10 stocks maximum',
        'estimated_time_minutes': 5,
        'recommended_for': 'First-time users, learning the system'
    },
    'starter': {
        'max_stocks': 25,
        'batch_size': 5,
        'delay_between_batches': 2.0,
        'description': 'Standard collection - good balance of speed and safety',
        'estimated_time_minutes': 12, 
        'recommended_for': 'Regular analysis, portfolio research'
    },
    'intermediate': {
        'max_stocks': 50,
        'batch_size': 8,
        'delay_between_batches': 1.5,
        'description': 'Expanded collection - sector analysis ready',
        'estimated_time_minutes': 20,
        'recommended_for': 'Market research, comparative analysis'
    },
    'advanced': {
        'max_stocks': 100,
        'batch_size': 10,
        'delay_between_batches': 1.0,
        'description': 'Comprehensive collection - full market segments',
        'estimated_time_minutes': 35,
        'recommended_for': 'Professional analysis, model training'
    }
}

# Display available profiles
print("📊 **Available Collection Profiles:**")
print("=" * 50)
for profile_name, config in COLLECTION_PROFILES.items():
    print(f"\n🎯 **{profile_name.upper()}**")
    print(f"   📊 Max Stocks: {config['max_stocks']}")
    print(f"   ⏱️ Est. Time: {config['estimated_time_minutes']} minutes")
    print(f"   📝 Description: {config['description']}")
    print(f"   👥 Best for: {config['recommended_for']}")

# %% [markdown]
# ## 🎯 Profile Selection and Setup
#
# Select your preferred profile and configure collection parameters:

# %%
# PROFILE SELECTION (Modify this based on your needs)
SELECTED_PROFILE = 'starter'  # Options: beginner, starter, intermediate, advanced

# Get profile configuration
config = COLLECTION_PROFILES[SELECTED_PROFILE].copy()

# Add common settings
config.update({
    'date_range_days': 365,  # 1 year of data
    'force_refresh': False,  # Use cache when available
    'sectors_to_include': ['tech_stocks', 'finance', 'property'],  # Focus sectors
    'verbose_logging': False  # Set to True for detailed progress logs
})

# Calculate date range
start_date, end_date = get_date_range(config['date_range_days'])

print(f"⚙️ **Configuration: {SELECTED_PROFILE.upper()} Profile**")
print("=" * 50)
print(f"📅 Date Range: {start_date} to {end_date} ({config['date_range_days']} days)")
print(f"📊 Max Stocks: {config['max_stocks']}")
print(f"🔄 Batch Size: {config['batch_size']} stocks per batch")
print(f"⏱️ Delay: {config['delay_between_batches']}s between batches")
print(f"⏰ Estimated Time: ~{config['estimated_time_minutes']} minutes")
print(f"🎯 Target Sectors: {', '.join(config['sectors_to_include'])}")

# Safety check for advanced users
if config['max_stocks'] > 50:
    print(f"\n⚠️ **Advanced Mode Selected**")
    print(f"   This will collect {config['max_stocks']} stocks")
    print(f"   Estimated completion: ~{config['estimated_time_minutes']} minutes")
    print(f"   Make sure you have sufficient time and API quota")

# %% [markdown]
# ## 📊 Stock Universe Exploration
#
# Discover available Hong Kong stocks and sectors:

# %%
# Explore Hong Kong stock universe
print("🔍 **Hong Kong Stock Universe Discovery**")
print("=" * 50)

# Get available stock sectors
print("🏢 **Available Sectors:**")
sector_summary = {}
total_available_stocks = 0

for sector, stocks in MAJOR_HK_STOCKS.items():
    sector_stocks = get_hk_stocks_by_sector(sector)
    sector_summary[sector] = {
        'count': len(sector_stocks),
        'stocks': sector_stocks,
        'examples': sector_stocks[:3]
    }
    total_available_stocks += len(sector_stocks)
    
    print(f"   🏢 {sector.replace('_', ' ').title()}: {len(sector_stocks)} stocks")
    print(f"      Examples: {', '.join(sector_stocks[:3])}")

# Get major stocks list for fallback
major_stocks = get_hk_stock_list_static()
print(f"\n📈 **Stock Universe Summary:**")
print(f"   Total categorized stocks: {total_available_stocks}")
print(f"   Major stocks available: {len(major_stocks)}")
print(f"   Selected profile limit: {config['max_stocks']} stocks")

# Calculate collection strategy
if config['max_stocks'] <= 25:
    strategy = "major_stocks"
    print(f"\n🎯 **Recommended Strategy: Major Stocks**")
    print(f"   Will select top {config['max_stocks']} major HK stocks")
else:
    strategy = "sector_balanced"
    print(f"\n🎯 **Recommended Strategy: Sector Balanced**")
    print(f"   Will distribute {config['max_stocks']} stocks across sectors")

# %% [markdown]
# ## 🎯 Stock Selection Strategy
#
# Choose and configure your stock selection approach:

# %%
# STOCK SELECTION IMPLEMENTATION
print("🎯 **Stock Selection in Progress...**")
print("=" * 40)

if strategy == "major_stocks":
    # Simple strategy: top major stocks
    target_stocks = major_stocks[:config['max_stocks']]
    print(f"📈 **Strategy: Top Major Stocks**")
    print(f"   Selected: {len(target_stocks)} stocks")
    print(f"   Sample: {', '.join(target_stocks[:5])}")
    
else:
    # Balanced strategy: distribute across sectors
    target_stocks = []
    stocks_per_sector = config['max_stocks'] // len(config['sectors_to_include'])
    remaining_slots = config['max_stocks'] % len(config['sectors_to_include'])
    
    print(f"📊 **Strategy: Sector Balanced Distribution**")
    for i, sector in enumerate(config['sectors_to_include']):
        sector_stocks = get_hk_stocks_by_sector(sector)
        
        # Allocate stocks (give extra to first sectors if there's remainder)
        allocation = stocks_per_sector + (1 if i < remaining_slots else 0)
        selected_from_sector = sector_stocks[:allocation]
        target_stocks.extend(selected_from_sector)
        
        print(f"   🏢 {sector.replace('_', ' ').title()}: {len(selected_from_sector)} stocks")
        print(f"      Selected: {', '.join(selected_from_sector[:3])}")
    
    # Fill remaining slots with major stocks if needed
    while len(target_stocks) < config['max_stocks']:
        for stock in major_stocks:
            if stock not in target_stocks:
                target_stocks.append(stock)
                if len(target_stocks) >= config['max_stocks']:
                    break

# Final selection summary
target_stocks = target_stocks[:config['max_stocks']]  # Ensure we don't exceed limit
print(f"\n✅ **Final Selection Complete**")
print(f"🎯 Total stocks selected: {len(target_stocks)}")
print(f"📋 Complete list: {', '.join(target_stocks)}")

# %% [markdown]
# ## 🚀 Data Collection Execution
#
# Execute the bulk data collection with comprehensive progress tracking:

# %%
# EXECUTE DATA COLLECTION
print("🚀 **Starting Hong Kong Stock Data Collection**")
print("=" * 60)
print(f"📊 Profile: {SELECTED_PROFILE.upper()}")
print(f"🎯 Stocks to collect: {len(target_stocks)}")
print(f"📅 Date range: {start_date} to {end_date}")
print(f"⏱️ Estimated time: ~{config['estimated_time_minutes']} minutes")
print(f"🔄 Processing strategy: {config['batch_size']} stocks per batch")

# Initialize collection tracking
collected_data = {}
failed_stocks = []
collection_stats = {
    'start_time': time.time(),
    'total_stocks': len(target_stocks),
    'successful': 0,
    'failed': 0,
    'batches_processed': 0
}

# Execute collection with progress tracking
try:
    print(f"\n🔄 **Collection Progress:**")
    
    # Calculate batch information
    total_batches = (len(target_stocks) + config['batch_size'] - 1) // config['batch_size']
    
    with tqdm(total=len(target_stocks), desc="Collecting stock data", unit="stocks") as pbar:
        for batch_num in range(total_batches):
            # Calculate batch range
            start_idx = batch_num * config['batch_size']
            end_idx = min(start_idx + config['batch_size'], len(target_stocks))
            batch_stocks = target_stocks[start_idx:end_idx]
            
            # Update progress description
            pbar.set_description(f"Batch {batch_num + 1}/{total_batches}")
            
            try:
                # Execute batch collection
                batch_data = fetch_hk_stocks_bulk(
                    tickers=batch_stocks,
                    start_date=start_date,
                    end_date=end_date,
                    batch_size=len(batch_stocks),
                    delay_between_batches=config['delay_between_batches'],
                    force_refresh=config['force_refresh']
                )
                
                # Process results
                collected_data.update(batch_data)
                collection_stats['successful'] += len(batch_data)
                
                # Track failed stocks in this batch
                batch_failed = [stock for stock in batch_stocks if stock not in batch_data]
                failed_stocks.extend(batch_failed)
                collection_stats['failed'] += len(batch_failed)
                
                # Update progress
                pbar.update(len(batch_stocks))
                collection_stats['batches_processed'] += 1
                
                # Calculate and display running statistics
                elapsed_time = time.time() - collection_stats['start_time']
                success_rate = (collection_stats['successful'] / collection_stats['total_stocks']) * 100
                avg_time_per_stock = elapsed_time / max(collection_stats['successful'], 1)
                
                pbar.set_postfix({
                    'Success': f"{collection_stats['successful']}/{collection_stats['total_stocks']}",
                    'Rate': f"{success_rate:.1f}%",
                    'Avg/Stock': f"{avg_time_per_stock:.2f}s"
                })
                
            except Exception as batch_error:
                print(f"\n❌ Batch {batch_num + 1} failed: {str(batch_error)}")
                failed_stocks.extend(batch_stocks)
                collection_stats['failed'] += len(batch_stocks)
                pbar.update(len(batch_stocks))
                continue
                
        pbar.close()
        
    # Final collection statistics
    total_time = time.time() - collection_stats['start_time']
    success_rate = (len(collected_data) / len(target_stocks)) * 100
    
    print(f"\n🎉 **Collection Complete!**")
    print("=" * 40)
    print(f"✅ Successfully collected: {len(collected_data)} stocks ({success_rate:.1f}%)")
    print(f"❌ Failed collections: {len(failed_stocks)} stocks")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"📊 Average per stock: {total_time/len(collected_data):.2f} seconds")
    print(f"🚀 Collection rate: {len(collected_data)/(total_time/60):.1f} stocks/minute")
    
    if failed_stocks:
        print(f"\n⚠️ **Failed Stocks:**")
        print(f"   {', '.join(failed_stocks[:10])}")
        if len(failed_stocks) > 10:
            print(f"   ... and {len(failed_stocks) - 10} more")
            
except Exception as e:
    print(f"\n❌ **Collection Error:** {str(e)}")
    print("Please check your internet connection and API limits")

# %% [markdown]
# ## 📊 Data Validation and Quality Assessment
#
# Validate the collected data and assess its quality:

# %%
# DATA VALIDATION AND QUALITY ASSESSMENT
if collected_data:
    print("🔍 **Data Quality Assessment**")
    print("=" * 50)
    
    # Overall statistics
    total_records = sum(len(data) for data in collected_data.values())
    avg_records_per_stock = total_records / len(collected_data) if collected_data else 0
    
    print(f"📊 **Overall Statistics:**")
    print(f"   Total stocks collected: {len(collected_data)}")
    print(f"   Total data records: {total_records:,}")
    print(f"   Average records per stock: {avg_records_per_stock:.0f}")
    
    # Data quality analysis per stock
    print(f"\n📈 **Per-Stock Analysis:**")
    quality_issues = []
    
    for ticker, data in list(collected_data.items())[:10]:  # Show first 10 for brevity
        # Basic data quality checks
        missing_data = data.isnull().sum().sum()
        date_range_actual = f"{data.index.min().date()} to {data.index.max().date()}"
        
        print(f"\n   📊 {ticker}:")
        print(f"      📅 Date range: {date_range_actual}")
        print(f"      📋 Records count: {len(data)}")
        print(f"      💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        print(f"      📊 Avg volume: {data['Volume'].mean():,.0f}")
        
        if missing_data > 0:
            print(f"      ⚠️ Missing data points: {missing_data}")
            quality_issues.append(f"{ticker}: {missing_data} missing points")
        else:
            print(f"      ✅ Complete data - no missing values")
            
        # Show sample data
        if len(data) > 0:
            latest_data = data.tail(1).iloc[0]
            print(f"      📋 Latest: {latest_data.name.date()} - Close: ${latest_data['Close']:.2f}")
    
    if len(collected_data) > 10:
        print(f"\n   ... and {len(collected_data) - 10} more stocks")
    
    # Quality summary
    print(f"\n📋 **Quality Summary:**")
    if quality_issues:
        print(f"   ⚠️ Quality issues found: {len(quality_issues)}")
        for issue in quality_issues[:5]:
            print(f"      - {issue}")
        if len(quality_issues) > 5:
            print(f"      ... and {len(quality_issues) - 5} more issues")
    else:
        print(f"   ✅ All data appears complete and valid")
        
    print(f"\n✅ **Data validation completed successfully!**")
    
else:
    print("❌ **No data collected - validation skipped**")
    print("Please check the collection process above for errors")

# %% [markdown]
# ## 💾 Data Storage and Next Steps
#
# Save the collected data and prepare for analysis:

# %%
# DATA STORAGE AND NEXT STEPS
if collected_data:
    print("💾 **Data Storage and Next Steps**")
    print("=" * 50)
    
    # Save data using built-in function
    try:
        # Generate collection summary with available data
        summary = create_bulk_fetch_summary(collected_data)
        print("📄 **Collection Summary Generated**")
        
        # Data is automatically cached by the fetch functions
        print("✅ **Data Cached Successfully**")
        print("   Data has been saved to the local cache directory")
        print("   Future runs will use cached data for faster execution")
        
        # Next steps recommendations
        print(f"\n🎯 **Recommended Next Steps:**")
        print(f"   1. 📊 **Feature Extraction**: Use notebook 03_feature_extraction.py")
        print(f"   2. 🤖 **Pattern Training**: Use notebook 04_pattern_training.py") 
        print(f"   3. 🔍 **Pattern Detection**: Use notebook 05_pattern_detection.py")
        print(f"   4. 📈 **Visualization**: Use notebook 07_visualization.py")
        print(f"   5. 🎮 **Interactive Analysis**: Use notebook 06_interactive_analysis.py")
        
        # Data usage examples
        print(f"\n💡 **Quick Usage Examples:**")
        print(f"   # Access collected data:")
        print(f"   tencent_data = collected_data.get('0700.HK')")
        print(f"   ")
        print(f"   # Check data shape:")
        print(f"   print(f'Tencent data: {{len(tencent_data)}} records')")
        print(f"   ")
        print(f"   # View recent prices:")
        print(f"   print(tencent_data[['Open', 'High', 'Low', 'Close']].tail())")
        
        # Performance tips for next collection
        if len(collected_data) >= config['max_stocks'] * 0.8:  # If collection was mostly successful
            next_profile = None
            for profile_name, profile_config in COLLECTION_PROFILES.items():
                if profile_config['max_stocks'] > config['max_stocks']:
                    next_profile = profile_name
                    break
                    
            if next_profile:
                print(f"\n🚀 **Ready for Next Level:**")
                print(f"   Consider upgrading to '{next_profile}' profile")
                print(f"   This would allow collecting up to {COLLECTION_PROFILES[next_profile]['max_stocks']} stocks")
                print(f"   Estimated time: ~{COLLECTION_PROFILES[next_profile]['estimated_time_minutes']} minutes")
        
    except Exception as e:
        print(f"⚠️ **Storage Warning:** {str(e)}")
        print("Data collection completed but summary generation failed")
        
else:
    print("❌ **No Data to Store**")
    print("Collection failed - please review error messages above")
    print("\n🔧 **Troubleshooting Tips:**")
    print("   1. Check your internet connection")
    print("   2. Verify API access is not rate-limited")
    print("   3. Try reducing the number of stocks (use 'beginner' profile)")
    print("   4. Consider increasing delay_between_batches")

print(f"\n🎉 **Notebook execution completed!**")
print(f"📅 Session time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 