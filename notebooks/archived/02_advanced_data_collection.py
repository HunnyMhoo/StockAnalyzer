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
# # 🚀 Advanced Hong Kong Stock Data Collection
#
# **Perfect for:** Intermediate/Advanced users, large-scale analysis (100-500+ stocks)
#
# This notebook provides **advanced techniques** for efficient, large-scale Hong Kong stock data collection.
#
# ## ⚡ Advanced Features
# - Parallel processing with configurable workers
# - Enterprise-grade rate limiting and retry logic
# - Checkpoint/resume capabilities for large operations
# - Memory-efficient streaming for 500+ stocks
# - Advanced error recovery and fault tolerance
# - Performance monitoring and optimization
#
# ## ⏱️ Time Estimates
# - **100 stocks**: 15-25 minutes
# - **200 stocks**: 30-45 minutes  
# - **500 stocks**: 1-2 hours
# - **1000+ stocks**: 3-4 hours (enterprise mode)

# %%
# Advanced setup with performance monitoring
from utilities.common_setup import setup_notebook, get_date_range, import_common_modules
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import concurrent.futures
import time
import psutil
import gc

# Initialize with performance tracking
print("🚀 Advanced Data Collection Setup")
print(f"💻 System Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"🖥️ CPU Cores: {psutil.cpu_count()}")

validation = setup_notebook()

# Import advanced collection modules
modules = import_common_modules()
get_hk_stock_list_static = modules['get_hk_stock_list_static']

# Import enterprise-grade functions
from bulk_collection_improved import (
    BulkCollectionConfig,
    BulkCollector,
    ResultsManager,
    create_enterprise_collector
)

from hk_stock_universe import (
    get_comprehensive_hk_stock_list,
    get_hk_stocks_by_sector,
    MAJOR_HK_STOCKS
)

print("✅ Advanced setup completed - Ready for large-scale collection!")

# %% [markdown]
# ## ⚙️ Advanced Configuration Profiles
#
# Choose from pre-configured profiles optimized for different scales:

# %%
# Define advanced configuration profiles
PROFILES = {
    'intermediate': {
        'max_stocks': 100,
        'batch_size': 10,
        'max_workers': 2,
        'delay_between_batches': 1.0,
        'enable_parallel': True,
        'checkpoint_every': 25,
        'memory_limit_gb': 4
    },
    
    'advanced': {
        'max_stocks': 300,
        'batch_size': 15,
        'max_workers': 3,
        'delay_between_batches': 0.8,
        'enable_parallel': True,
        'checkpoint_every': 50,
        'memory_limit_gb': 8
    },
    
    'enterprise': {
        'max_stocks': 1000,
        'batch_size': 20,
        'max_workers': 4,
        'delay_between_batches': 0.5,
        'enable_parallel': True,
        'checkpoint_every': 100,
        'memory_limit_gb': 16
    }
}

# Select your profile (modify as needed)
SELECTED_PROFILE = 'intermediate'  # Options: intermediate, advanced, enterprise

config = PROFILES[SELECTED_PROFILE].copy()
config.update({
    'date_range_days': 365,
    'force_refresh': False,
    'enable_resume': True,
    'save_checkpoints': True
})

print(f"⚙️ **Selected Profile: {SELECTED_PROFILE.upper()}**")
print("=" * 50)
for key, value in config.items():
    print(f"   {key}: {value}")

# Calculate date range
start_date, end_date = get_date_range(config['date_range_days'])
print(f"\n📅 **Date Range:** {start_date} to {end_date}")

# Memory check
current_memory = psutil.virtual_memory()
if current_memory.available / (1024**3) < config['memory_limit_gb']:
    print(f"⚠️ **Memory Warning:** Available memory ({current_memory.available / (1024**3):.1f} GB) < recommended ({config['memory_limit_gb']} GB)")
    print("Consider reducing max_stocks or using a lower profile")

# %% [markdown]
# ## 🎯 Advanced Stock Universe Discovery
#
# Discover and categorize the complete Hong Kong stock universe:

# %%
# Advanced stock universe discovery
print("🔍 **Hong Kong Stock Universe Discovery**")
print("=" * 50)

# Get comprehensive stock list
try:
    comprehensive_stocks = get_comprehensive_hk_stock_list()
    print(f"📈 **Comprehensive Universe:** {len(comprehensive_stocks)} stocks")
except Exception as e:
    print(f"⚠️ Using fallback static list: {e}")
    comprehensive_stocks = get_hk_stock_list_static()

# Categorize by sectors
sector_analysis = {}
total_sector_stocks = 0

for sector, stocks in MAJOR_HK_STOCKS.items():
    sector_stocks = get_hk_stocks_by_sector(sector)
    sector_analysis[sector] = {
        'count': len(sector_stocks),
        'stocks': sector_stocks,
        'percentage': len(sector_stocks) / len(comprehensive_stocks) * 100
    }
    total_sector_stocks += len(sector_stocks)

print(f"\n📊 **Sector Analysis:**")
for sector, data in sector_analysis.items():
    print(f"   🏢 {sector.upper()}: {data['count']} stocks ({data['percentage']:.1f}%)")

print(f"\n📋 **Universe Statistics:**")
print(f"   Total discovered: {len(comprehensive_stocks)}")
print(f"   Categorized in sectors: {total_sector_stocks}")
print(f"   Uncategorized: {len(comprehensive_stocks) - total_sector_stocks}")

# Select target universe based on profile
if config['max_stocks'] >= len(comprehensive_stocks):
    target_universe = comprehensive_stocks
    print(f"\n🎯 **Target: Full Universe** ({len(target_universe)} stocks)")
else:
    # Prioritize major stocks and sector diversity
    priority_stocks = []
    
    # Add major stocks first
    major_stocks = get_hk_stock_list_static()
    priority_stocks.extend(major_stocks[:config['max_stocks']//2])
    
    # Add sector diversity
    remaining_slots = config['max_stocks'] - len(priority_stocks)
    stocks_per_sector = remaining_slots // len(sector_analysis)
    
    for sector, data in sector_analysis.items():
        sector_stocks = [s for s in data['stocks'] if s not in priority_stocks]
        priority_stocks.extend(sector_stocks[:stocks_per_sector])
    
    target_universe = priority_stocks[:config['max_stocks']]
    print(f"\n🎯 **Target: Curated Selection** ({len(target_universe)} stocks)")

print(f"📋 **Sample targets:** {', '.join(target_universe[:10])}")

# %% [markdown]
# ## 🔄 Advanced Collection with Parallel Processing
#
# Execute large-scale collection with enterprise features:

# %%
# Initialize advanced collector
print("🚀 **Initializing Advanced Collection Engine**")
print("=" * 50)

# Create enterprise collector with selected configuration
collector_config = BulkCollectionConfig()
collector_config.max_workers = config['max_workers']
collector_config.batch_size = config['batch_size']
collector_config.delay_between_batches = config['delay_between_batches']
collector_config.enable_parallel = config['enable_parallel']

# Initialize collector
advanced_collector = create_enterprise_collector(collector_config)

# Performance monitoring setup
start_time = time.time()
initial_memory = psutil.virtual_memory().used / (1024**3)

print(f"⚙️ **Collection Configuration:**")
print(f"   Parallel processing: {'✅ Enabled' if config['enable_parallel'] else '❌ Disabled'}")
print(f"   Workers: {config['max_workers']}")
print(f"   Batch size: {config['batch_size']}")
print(f"   Checkpoint every: {config['checkpoint_every']} stocks")
print(f"   Estimated time: {len(target_universe) * config['delay_between_batches'] / (config['max_workers'] * 60):.1f} minutes")

# Memory monitoring function
def monitor_memory():
    current = psutil.virtual_memory()
    used_gb = current.used / (1024**3)
    available_gb = current.available / (1024**3)
    print(f"💾 Memory: {used_gb:.1f}GB used, {available_gb:.1f}GB available ({current.percent:.1f}%)")

monitor_memory()

# %% [markdown]
# ## 🎯 Execute Advanced Collection
#
# Run the large-scale collection with checkpointing:

# %%
# Execute advanced collection with checkpointing
print("🚀 **Starting Advanced Bulk Collection**")
print("=" * 60)

collection_results = {}
checkpoint_data = {}
failed_stocks = []

try:
    # Configure logging verbosity
    VERBOSE_LOGGING = False  # Set to True for detailed batch logging
    SHOW_MEMORY = True      # Set to False to hide memory monitoring
    
    # Process in checkpointed batches with smart progress tracking
    total_batches = (len(target_universe) + config['checkpoint_every'] - 1) // config['checkpoint_every']
    
    print(f"\n🔄 Processing {len(target_universe)} stocks in {total_batches} checkpointed batches...")
    
    with tqdm(total=len(target_universe), desc="Enterprise Collection", unit="stocks") as pbar:
        for i in range(0, len(target_universe), config['checkpoint_every']):
            batch_end = min(i + config['checkpoint_every'], len(target_universe))
            current_batch = target_universe[i:batch_end]
            batch_num = i//config['checkpoint_every'] + 1
            
            if VERBOSE_LOGGING:
                print(f"\n📦 **Processing Batch {batch_num}** ({len(current_batch)} stocks)")
                print(f"   Range: {i+1} to {batch_end} of {len(target_universe)}")
            
            # Monitor memory before batch (only if enabled)
            if SHOW_MEMORY and VERBOSE_LOGGING:
                monitor_memory()
            
            # Process current batch
            try:
                batch_results = advanced_collector.fetch_stocks_parallel(
                    tickers=current_batch,
                    start_date=start_date,
                    end_date=end_date,
                    verbose=False  # Suppress per-stock logs
                )
                
                # Merge results
                collection_results.update(batch_results)
                
                # Calculate batch statistics
                batch_success = len(batch_results)
                batch_failed = len(current_batch) - batch_success
                success_rate = batch_success / len(current_batch) * 100
                
                if VERBOSE_LOGGING:
                    print(f"   ✅ Batch completed: {batch_success}/{len(current_batch)} ({success_rate:.1f}% success)")
                
                # Track failed stocks (quietly unless verbose)
                if batch_failed > 0:
                    failed_batch = [stock for stock in current_batch if stock not in batch_results]
                    failed_stocks.extend(failed_batch)
                    if VERBOSE_LOGGING and batch_failed <= 5:
                        print(f"   ❌ Failed stocks: {', '.join(failed_batch)}")
                    elif VERBOSE_LOGGING:
                        print(f"   ❌ Failed: {batch_failed} stocks")
                
                # Save checkpoint (quietly)
                if config['save_checkpoints']:
                    checkpoint_data = {
                        'completed_stocks': list(collection_results.keys()),
                        'failed_stocks': failed_stocks,
                        'batch_number': batch_num,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                # Force garbage collection after each batch
                gc.collect()
                
                # Update progress bar with comprehensive info
                total_completed = len(collection_results)
                overall_progress = total_completed / len(target_universe) * 100
                elapsed_time = time.time() - start_time
                
                # Calculate ETA
                if total_completed > 0:
                    eta_minutes = (elapsed_time / total_completed) * (len(target_universe) - total_completed) / 60
                    pbar.set_postfix({
                        'Batch': f"{batch_num}/{total_batches}",
                        'Success': f"{success_rate:.0f}%",
                        'Overall': f"{overall_progress:.1f}%",
                        'ETA': f"{eta_minutes:.0f}m"
                    })
                
                pbar.update(len(current_batch))
                
                # Show progress summary every 5 batches (if verbose)
                if VERBOSE_LOGGING and batch_num % 5 == 0:
                    print(f"   📊 Overall progress: {total_completed}/{len(target_universe)} ({overall_progress:.1f}%)")
                    print(f"   ⏱️ Elapsed: {elapsed_time/60:.1f}m, ETA: {eta_minutes:.1f}m")
                
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"   ❌ Batch error: {e}")
                failed_stocks.extend(current_batch)
                pbar.update(len(current_batch))
                continue
    
    # Print comprehensive collection summary
    from common_setup import print_collection_summary
    print_collection_summary(
        collected_data=collection_results,
        failed_stocks=failed_stocks,
        target_count=len(target_universe),
        start_time=start_time,
        show_failed_details=True,
        max_failed_shown=15  # Show more details for advanced users
    )
    
except KeyboardInterrupt:
    print(f"\n⏸️ **Collection Interrupted**")
    print(f"   Partial results available: {len(collection_results)} stocks")
    
    # Show interrupted collection summary
    from common_setup import print_collection_summary
    print_collection_summary(
        collected_data=collection_results,
        failed_stocks=failed_stocks,
        target_count=len(target_universe),
        start_time=start_time,
        show_failed_details=True,
        max_failed_shown=5
    )
    
except Exception as e:
    print(f"\n❌ **Collection Error:** {e}")

# Additional performance metrics for advanced users
total_time = time.time() - start_time
final_memory = psutil.virtual_memory().used / (1024**3)
memory_used = final_memory - initial_memory

print(f"\n💻 **Advanced Performance Metrics:**")
print(f"   💾 Memory used: {memory_used:.1f} GB")
print(f"   🧠 Memory per stock: {memory_used/len(collection_results):.3f} GB" if len(collection_results) > 0 else "   🧠 Memory per stock: N/A")
print(f"   ⚡ Peak memory efficiency: {len(collection_results)/memory_used:.1f} stocks/GB" if memory_used > 0 else "   ⚡ Peak memory efficiency: N/A")

# %% [markdown]
# ## 📊 Advanced Data Quality Analysis
#
# Comprehensive analysis of collected data quality and performance:

# %%
# Advanced quality analysis
if collection_results:
    print("📊 **Advanced Data Quality Analysis**")
    print("=" * 60)
    
    # Create comprehensive summary
    summary_data = []
    total_records = 0
    
    for ticker, data in collection_results.items():
        if data is not None and not data.empty:
            record_count = len(data)
            total_records += record_count
            
            # Calculate quality metrics
            completeness = record_count / 252 * 100  # Assume 252 trading days per year
            price_range = data['Close'].max() - data['Close'].min()
            volatility = data['Close'].std()
            avg_volume = data['Volume'].mean()
            
            summary_data.append({
                'Ticker': ticker,
                'Records': record_count,
                'Completeness': f"{completeness:.1f}%",
                'Price_Range': f"${price_range:.2f}",
                'Volatility': f"{volatility:.2f}",
                'Avg_Volume': f"{avg_volume:,.0f}",
                'Start_Date': data.index[0].date(),
                'End_Date': data.index[-1].date()
            })
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame(summary_data)
    
    print(f"📈 **Overall Statistics:**")
    print(f"   Total stocks analyzed: {len(analysis_df)}")
    print(f"   Total records: {total_records:,}")
    print(f"   Average records per stock: {total_records/len(analysis_df):.0f}")
    print(f"   Data efficiency: {len(collection_results)/len(target_universe)*100:.1f}%")
    
    # Quality distribution
    record_counts = [int(r) for r in analysis_df['Records']]
    quality_tiers = {
        'High Quality (>200 records)': sum(1 for r in record_counts if r > 200),
        'Medium Quality (100-200 records)': sum(1 for r in record_counts if 100 <= r <= 200),
        'Low Quality (<100 records)': sum(1 for r in record_counts if r < 100)
    }
    
    print(f"\n🎯 **Quality Distribution:**")
    for tier, count in quality_tiers.items():
        percentage = count / len(analysis_df) * 100
        print(f"   {tier}: {count} stocks ({percentage:.1f}%)")
    
    # Show top performers
    analysis_df['Records_Int'] = record_counts
    top_quality = analysis_df.nlargest(10, 'Records_Int')
    
    print(f"\n🏆 **Top 10 Highest Quality Stocks:**")
    print(top_quality[['Ticker', 'Records', 'Completeness']].to_string(index=False))
    
    # Failed stocks analysis
    if failed_stocks:
        print(f"\n⚠️ **Failed Stocks Analysis:**")
        print(f"   Total failed: {len(failed_stocks)}")
        print(f"   Failure rate: {len(failed_stocks)/len(target_universe)*100:.1f}%")
        print(f"   Sample failed: {', '.join(failed_stocks[:10])}")
        
else:
    print("❌ No data available for analysis")

# %% [markdown]
# ## 💾 Enterprise Data Management
#
# Save data with enterprise-grade organization and metadata:

# %%
# Enterprise data management
if collection_results:
    print("💾 **Enterprise Data Management**")
    print("=" * 50)
    
    # Create results manager
    results_manager = ResultsManager()
    
    # Generate comprehensive metadata
    collection_metadata = {
        'collection_profile': SELECTED_PROFILE,
        'configuration': config,
        'collection_statistics': {
            'total_targeted': len(target_universe),
            'successfully_collected': len(collection_results),
            'failed_stocks': len(failed_stocks),
            'success_rate': len(collection_results)/len(target_universe)*100,
            'total_records': sum(len(data) for data in collection_results.values() if data is not None),
            'collection_time_minutes': total_time/60,
            'performance_stocks_per_minute': len(collection_results)/(total_time/60)
        },
        'date_range': {
            'start_date': start_date,
            'end_date': end_date,
            'collection_date': datetime.now().isoformat()
        },
        'system_info': {
            'memory_used_gb': memory_used,
            'cpu_cores': psutil.cpu_count(),
            'parallel_workers': config['max_workers']
        }
    }
    
    try:
        # Save with enterprise structure
        saved_files = results_manager.save_enterprise_collection(
            stock_data=collection_results,
            metadata=collection_metadata,
            base_name=f"advanced_collection_{SELECTED_PROFILE}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        
        print("✅ **Enterprise save completed!**")
        print(f"📁 **Generated files:**")
        for file_info in saved_files:
            print(f"   • {file_info['path']} ({file_info['size']} MB)")
            
        # Save checkpoint for future resume
        if config['save_checkpoints']:
            checkpoint_file = f"checkpoint_{SELECTED_PROFILE}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(checkpoint_file, 'w') as f:
                import json
                json.dump(checkpoint_data, f, indent=2)
            print(f"   • {checkpoint_file} (checkpoint)")
            
    except Exception as e:
        print(f"❌ **Save error:** {e}")
        print("Data remains available in memory as 'collection_results'")

else:
    print("⚠️ No data to save")

# %% [markdown]
# ## 📈 Performance Analysis & Optimization
#
# Analyze collection performance and provide optimization recommendations:

# %%
# Performance analysis and optimization recommendations
print("📈 **Performance Analysis & Optimization**")
print("=" * 60)

if collection_results:
    # Calculate performance metrics
    stocks_per_minute = len(collection_results) / (total_time / 60)
    records_per_minute = total_records / (total_time / 60)
    efficiency_score = len(collection_results) / len(target_universe) * 100
    
    print(f"⚡ **Performance Metrics:**")
    print(f"   Stocks per minute: {stocks_per_minute:.1f}")
    print(f"   Records per minute: {records_per_minute:,.0f}")
    print(f"   Collection efficiency: {efficiency_score:.1f}%")
    print(f"   Memory efficiency: {memory_used/len(collection_results):.3f} GB per stock")
    
    # Performance tier assessment
    if stocks_per_minute > 20:
        perf_tier = "🚀 Excellent"
    elif stocks_per_minute > 10:
        perf_tier = "✅ Good"
    elif stocks_per_minute > 5:
        perf_tier = "⚠️ Moderate"
    else:
        perf_tier = "❌ Slow"
    
    print(f"   Overall performance: {perf_tier}")
    
    # Optimization recommendations
    print(f"\n🎯 **Optimization Recommendations:**")
    
    if efficiency_score < 90:
        print("   🔧 Increase max_retries for better success rate")
        print("   🔧 Add delay_between_requests to reduce API errors")
    
    if stocks_per_minute < 10:
        print("   ⚡ Consider reducing delay_between_batches")
        print("   ⚡ Increase max_workers (if system resources allow)")
    
    if memory_used > 8:
        print("   💾 Enable memory cleanup between batches")
        print("   💾 Reduce checkpoint_every for more frequent cleanup")
    
    if len(failed_stocks) > len(target_universe) * 0.1:
        print("   🛡️ Implement retry logic for failed stocks")
        print("   🛡️ Add exponential backoff for rate limiting")
    
    # Scaling recommendations
    print(f"\n📊 **Scaling Recommendations:**")
    
    if SELECTED_PROFILE == 'intermediate' and efficiency_score > 85:
        print("   🚀 Ready to upgrade to 'advanced' profile")
        print("   🚀 Can handle 300+ stocks efficiently")
    
    if SELECTED_PROFILE == 'advanced' and efficiency_score > 85:
        print("   🏢 Ready for 'enterprise' profile")
        print("   🏢 Can scale to 1000+ stocks")
    
    # Next steps
    print(f"\n✅ **Next Steps:**")
    print("   1. Use collected data in '04_feature_extraction.ipynb'")
    print("   2. Scale to pattern recognition with '05_pattern_model_training.ipynb'")
    print("   3. Analyze patterns with '07_pattern_match_visualization.ipynb'")
    print("   4. Set up monitoring with '08_signal_outcome_tagging.ipynb'")

else:
    print("❌ No performance data available")

print(f"\n📅 **Advanced collection completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ---
# **🚀 Advanced Hong Kong Stock Data Collection**  
# *Enterprise-grade bulk collection with parallel processing - 100+ stocks* 
