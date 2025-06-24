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
# # 🔰 Basic Hong Kong Stock Data Collection
#
# **Perfect for:** Beginners, small-scale analysis, testing (10-50 stocks)
#
# This notebook provides a **simple, reliable approach** for collecting Hong Kong stock data in batches.
#
# ## ✅ What You'll Learn
# - Fetch 10-50 stocks safely with built-in rate limiting
# - Sector-based stock selection (Tech, Finance, Property)
# - Progress tracking and error handling
# - Data validation and quality checks
# - Safe caching and incremental updates
#
# ## ⏱️ Time Required
# - **Setup**: 2 minutes
# - **Small batch (10 stocks)**: 3-5 minutes  
# - **Medium batch (25 stocks)**: 8-12 minutes
# - **Large batch (50 stocks)**: 15-20 minutes

# %%
# Setup using shared utilities
from common_setup import setup_notebook, get_date_range, import_common_modules
from datetime import datetime, timedelta
import pandas as pd
import time
from tqdm.notebook import tqdm

# Initialize notebook environment
print("🔰 Basic Data Collection Setup")
validation = setup_notebook()

# Import data collection modules
modules = import_common_modules()
get_hk_stock_list_static = modules['get_hk_stock_list_static']

# Import specific bulk collection functions
from bulk_data_fetcher import (
    fetch_hk_stocks_bulk,
    fetch_all_major_hk_stocks,
    create_bulk_fetch_summary,
    save_bulk_data
)

from hk_stock_universe import get_hk_stocks_by_sector, MAJOR_HK_STOCKS

print("✅ Setup completed - Ready for basic bulk collection!")

# %% [markdown]
# ## 📊 Step 1: Explore Available Stock Categories
#
# Let's see what stock categories are available for collection:

# %%
# Explore available stock sectors
print("📊 **Available HK Stock Sectors:**")
print("=" * 50)

sector_info = []
for sector, stocks in MAJOR_HK_STOCKS.items():
    sector_info.append({
        'Sector': sector.replace('_', ' ').title(),
        'Count': len(stocks),
        'Examples': ', '.join(stocks[:3])
    })
    print(f"🏢 {sector.upper()}: {len(stocks)} stocks")
    print(f"   Examples: {', '.join(stocks[:3])}")

# Get total available stocks
all_stocks = get_hk_stock_list_static()
print(f"\n📈 **Total Available Stocks:** {len(all_stocks)}")
print(f"🎯 **Recommended for beginners:** Start with 10-25 stocks")

# %% [markdown]
# ## ⚙️ Step 2: Configure Collection Parameters
#
# Set up your data collection preferences:

# %%
# Configuration - Modify these settings as needed
COLLECTION_CONFIG = {
    'date_range_days': 365,  # 1 year of data
    'batch_size': 5,         # Process 5 stocks at a time (safe for beginners)
    'delay_between_batches': 2.0,  # 2 seconds between batches (conservative)
    'max_stocks': 25,        # Limit for safety
    'sectors_to_include': ['tech_stocks', 'finance', 'property'],  # Focus sectors
    'force_refresh': False   # Use cache when available
}

# Calculate date range
start_date, end_date = get_date_range(COLLECTION_CONFIG['date_range_days'])

print("⚙️ **Collection Configuration:**")
print(f"📅 Date Range: {start_date} to {end_date}")
print(f"📊 Max Stocks: {COLLECTION_CONFIG['max_stocks']}")
print(f"🔄 Batch Size: {COLLECTION_CONFIG['batch_size']}")
print(f"⏱️ Delay: {COLLECTION_CONFIG['delay_between_batches']}s")
print(f"🎯 Target Sectors: {', '.join(COLLECTION_CONFIG['sectors_to_include'])}")

# %% [markdown]
# ## 🎯 Step 3: Select Stocks for Collection
#
# Choose from different selection strategies:

# %%
# Strategy 1: Top Major Stocks (Recommended for beginners)
print("🎯 **Stock Selection Strategies:**")
print("=" * 40)

# Get major stocks (most reliable)
major_stocks = get_hk_stock_list_static()[:COLLECTION_CONFIG['max_stocks']]

# Get sector-specific stocks
tech_stocks = get_hk_stocks_by_sector('tech_stocks')
finance_stocks = get_hk_stocks_by_sector('finance') 
property_stocks = get_hk_stocks_by_sector('property')

print(f"📈 **Strategy 1 - Major Stocks:** {len(major_stocks)} stocks")
print(f"   Examples: {', '.join(major_stocks[:5])}")

print(f"\n💻 **Strategy 2 - Tech Focus:** {len(tech_stocks)} stocks")
print(f"   Examples: {', '.join(tech_stocks[:3])}")

print(f"\n🏦 **Strategy 3 - Finance Focus:** {len(finance_stocks)} stocks") 
print(f"   Examples: {', '.join(finance_stocks[:3])}")

print(f"\n🏢 **Strategy 4 - Property Focus:** {len(property_stocks)} stocks")
print(f"   Examples: {', '.join(property_stocks[:3])}")

# Choose your strategy (modify as needed)
SELECTED_STRATEGY = "major_stocks"  # Options: major_stocks, tech, finance, property

if SELECTED_STRATEGY == "major_stocks":
    target_stocks = major_stocks
elif SELECTED_STRATEGY == "tech":
    target_stocks = tech_stocks[:COLLECTION_CONFIG['max_stocks']]
elif SELECTED_STRATEGY == "finance":
    target_stocks = finance_stocks[:COLLECTION_CONFIG['max_stocks']]
elif SELECTED_STRATEGY == "property":
    target_stocks = property_stocks[:COLLECTION_CONFIG['max_stocks']]
else:
    target_stocks = major_stocks

print(f"\n✅ **Selected Strategy:** {SELECTED_STRATEGY}")
print(f"🎯 **Target Stocks:** {len(target_stocks)} stocks")
print(f"📋 **Stock List:** {', '.join(target_stocks)}")

# %% [markdown]
# ## 🚀 Step 4: Execute Data Collection
#
# Run the bulk data collection with progress tracking:

# %%
# Execute bulk data collection with smart logging
print("🚀 **Starting Basic Bulk Data Collection**")
print("=" * 50)
print(f"📊 Collecting {len(target_stocks)} stocks")
print(f"📅 Date range: {start_date} to {end_date}")
print(f"⏱️ Estimated time: {len(target_stocks) * COLLECTION_CONFIG['delay_between_batches'] / 60:.1f} minutes")

# Configure logging level (set to False to reduce output)
VERBOSE_LOGGING = False  # Set to True for detailed per-stock logging

# Run the collection with progress tracking
collected_data = {}
failed_stocks = []
start_time = time.time()  # Track collection time

try:
    print(f"\n🔄 Processing {len(target_stocks)} stocks in batches of {COLLECTION_CONFIG['batch_size']}...")
    
    # Process stocks with progress bar instead of individual logs
    with tqdm(total=len(target_stocks), desc="Collecting data", unit="stocks") as pbar:
        for i in range(0, len(target_stocks), COLLECTION_CONFIG['batch_size']):
            batch = target_stocks[i:i + COLLECTION_CONFIG['batch_size']]
            
            # Process batch quietly
            try:
                batch_data = fetch_hk_stocks_bulk(
                    tickers=batch,
                    start_date=start_date,
                    end_date=end_date,
                    batch_size=len(batch),
                    delay_between_batches=COLLECTION_CONFIG['delay_between_batches'],
                    force_refresh=COLLECTION_CONFIG['force_refresh'],
                    verbose=VERBOSE_LOGGING  # Control verbosity
                )
                
                # Merge successful results
                collected_data.update(batch_data)
                
                # Track failed stocks (quietly)
                batch_failed = [stock for stock in batch if stock not in batch_data]
                failed_stocks.extend(batch_failed)
                
                # Update progress bar with summary
                success_count = len(batch_data)
                pbar.set_postfix({
                    'Success': f"{len(collected_data)}/{len(target_stocks)}", 
                    'Batch': f"{success_count}/{len(batch)}"
                })
                pbar.update(len(batch))
                
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"   ⚠️ Batch error: {e}")
                failed_stocks.extend(batch)
                pbar.update(len(batch))
                continue
    
    # Print comprehensive summary
    from common_setup import print_collection_summary
    print_collection_summary(
        collected_data=collected_data,
        failed_stocks=failed_stocks,
        target_count=len(target_stocks),
        start_time=start_time,
        show_failed_details=True,
        max_failed_shown=5  # Keep it brief for basic collection
    )
    
except Exception as e:
    print(f"❌ **Collection Error:** {e}")
    collected_data = {}
    failed_stocks = target_stocks

# %% [markdown]
# ## 📊 Step 5: Data Quality Analysis
#
# Analyze the collected data quality and completeness:

# %%
# Analyze collected data quality
if collected_data:
    print("📊 **Data Quality Analysis**")
    print("=" * 50)
    
    # Create summary
    summary_df = create_bulk_fetch_summary(collected_data)
    
    print(f"📈 **Overall Statistics:**")
    print(f"   Total stocks collected: {len(summary_df)}")
    print(f"   Total records: {summary_df['Records'].sum():,}")
    print(f"   Average records per stock: {summary_df['Records'].mean():.0f}")
    print(f"   Date range: {summary_df['Start Date'].min()} to {summary_df['End Date'].max()}")
    
    # Quality metrics
    avg_completeness = summary_df['Records'].mean()
    min_records = summary_df['Records'].min()
    max_records = summary_df['Records'].max()
    
    print(f"\n🎯 **Quality Metrics:**")
    print(f"   Minimum records: {min_records}")
    print(f"   Maximum records: {max_records}")
    print(f"   Completeness ratio: {min_records/max_records:.2%}")
    
    # Show detailed summary
    print(f"\n📋 **Detailed Summary:**")
    print(summary_df.to_string(index=False))
    
    # Check for potential issues
    low_quality_stocks = summary_df[summary_df['Records'] < avg_completeness * 0.8]
    if not low_quality_stocks.empty:
        print(f"\n⚠️ **Stocks with lower data quality:**")
        print(low_quality_stocks[['Ticker', 'Records']].to_string(index=False))
    else:
        print(f"\n✅ **All stocks have good data quality!**")
        
else:
    print("❌ No data collected - check your configuration and try again")

# %% [markdown]
# ## 💾 Step 6: Save Collected Data
#
# Save the data with proper organization:

# %%
# Save collected data
if collected_data:
    print("💾 **Saving Collected Data**")
    print("=" * 40)
    
    # Save using the bulk data saver
    try:
        saved_files = save_bulk_data(
            stock_data=collected_data,
            base_filename=f"basic_collection_{datetime.now().strftime('%Y%m%d')}"
        )
        
        print("✅ **Data saved successfully!**")
        print(f"📁 **Files created:**")
        for file_path in saved_files:
            file_size = len(str(file_path))  # Rough estimate
            print(f"   • {file_path}")
            
    except Exception as e:
        print(f"❌ **Save Error:** {e}")
        print("Data is still available in memory as 'collected_data'")
        
else:
    print("⚠️ No data to save")

# %% [markdown]
# ## 📈 Step 7: Quick Data Preview
#
# Preview some of the collected data:

# %%
# Preview collected data
if collected_data:
    print("📈 **Data Preview**")
    print("=" * 40)
    
    # Show first stock as example
    first_ticker = list(collected_data.keys())[0]
    first_data = collected_data[first_ticker]
    
    print(f"📊 **Sample Data for {first_ticker}:**")
    print(f"   Records: {len(first_data)}")
    print(f"   Columns: {list(first_data.columns)}")
    print(f"   Date range: {first_data.index[0].date()} to {first_data.index[-1].date()}")
    
    print(f"\n📋 **Recent data sample:**")
    print(first_data.tail().round(2))
    
    # Basic statistics
    print(f"\n📊 **Price Statistics for {first_ticker}:**")
    print(f"   Close price range: ${first_data['Close'].min():.2f} - ${first_data['Close'].max():.2f}")
    print(f"   Average volume: {first_data['Volume'].mean():,.0f}")
    print(f"   Volatility (std): {first_data['Close'].std():.2f}")
    
else:
    print("❌ No data available for preview")

# %% [markdown]
# ## ✅ Collection Summary & Next Steps
#
# Review what was accomplished and suggest next steps:

# %%
# Final summary
print("🎉 **Basic Data Collection Summary**")
print("=" * 50)

if collected_data:
    total_records = sum(len(data) for data in collected_data.values())
    
    print(f"✅ **Success!** Collected data for {len(collected_data)} stocks")
    print(f"📊 Total records: {total_records:,}")
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"⏱️ Collection strategy: {SELECTED_STRATEGY}")
    
    print(f"\n🚀 **Next Steps:**")
    print(f"   1. Use '04_feature_extraction.ipynb' to extract technical indicators")
    print(f"   2. Try '05_pattern_model_training.ipynb' for ML model training")
    print(f"   3. Explore '07_pattern_match_visualization.ipynb' for charts")
    print(f"   4. Scale up with '02_advanced_data_collection.ipynb'")
    
    print(f"\n🎯 **Tips for Next Time:**")
    print(f"   • Increase max_stocks to 50-100 for more comprehensive analysis")
    print(f"   • Try different sectors (tech_stocks, finance, property)")
    print(f"   • Reduce delay_between_batches to 1.0s for faster collection")
    print(f"   • Use force_refresh=True to get the latest data")
    
else:
    print("❌ **Collection failed - Troubleshooting:**")
    print("   1. Check internet connection")
    print("   2. Verify ticker symbols are valid (.HK format)")
    print("   3. Try reducing max_stocks to 10")
    print("   4. Increase delay_between_batches to 3.0s")

print(f"\n📅 **Collection completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ---
# **🔰 Basic Hong Kong Stock Data Collection**  
# *Simple, reliable bulk data collection for beginners - up to 50 stocks* 
