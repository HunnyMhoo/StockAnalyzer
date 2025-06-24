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
# # Hong Kong Stock Bulk Data Collection - Comprehensive Guide
#
# This notebook provides **efficient, production-ready approaches** for bulk fetching Hong Kong stock data.
#
# ## 📋 **What You'll Master**
#
# | Level | Approach | Best For |
# |-------|----------|----------|
# | 🔰 **Beginner** | Curated Stock Lists | Learning & Testing (10-50 stocks) |
# | 📊 **Intermediate** | Sector-Based Fetching | Targeted Analysis (50-200 stocks) |
# | 🚀 **Advanced** | Full Universe Discovery | Comprehensive Research (500+ stocks) |
# | ⚡ **Enterprise** | Parallel Processing | Production Systems (1000+ stocks) |
#
# ## 🎯 **Key Features**
# - **Smart Rate Limiting**: Respects API constraints
# - **Error Recovery**: Robust retry logic with exponential backoff
# - **Progress Tracking**: Real-time monitoring for large operations
# - **Data Management**: Systematic saving and organization
# - **Memory Efficient**: Batch processing to handle large datasets
#
# ---
# **📝 Note**: Run the Setup section first, then choose your preferred approach.
#

# %% [raw]
# ## 🔧 **Setup & Configuration**
#
# **Run this section first** - All subsequent cells depend on this setup.
#

# %%
# Standard Library Imports
import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup Python Path
notebook_dir = Path().absolute()
src_dir = notebook_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

print("✅ Standard libraries loaded")
print(f"📁 Source path: {src_dir}")
print(f"📍 Working directory: {notebook_dir}")

# Verify setup
if src_dir.exists():
    print("🎯 Setup successful - ready to proceed!")
else:
    print("⚠️ Source directory not found - check your notebook location")


# %%
# Import Custom Modules and Improved Utilities
try:
    # Core stock data modules
    from hk_stock_universe import (
        get_hk_stock_list_static,
        get_hk_stocks_by_sector,
        get_comprehensive_hk_stock_list,
        MAJOR_HK_STOCKS
    )
    
    from bulk_data_fetcher import (
        fetch_hk_stocks_bulk,
        fetch_all_major_hk_stocks,
        fetch_hk_tech_stocks,
        create_bulk_fetch_summary,
        save_bulk_data
    )
    
    # Import the improved utilities we created
    from bulk_collection_improved import (
        BulkCollectionConfig,
        BulkCollector,
        ResultsManager,
        create_beginner_collector,
        create_enterprise_collector,
        quick_demo
    )
    
    print("✅ All modules imported successfully!")
    print("🚀 Enhanced utilities loaded - ready for efficient bulk collection!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n🔧 Troubleshooting:")
    print("   1. Ensure you're running from the notebooks/ directory")
    print("   2. Check that all source files exist in ../src/")
    print("   3. Verify Python path setup in previous cell")
    raise


# %%
# Global Configuration - Centralized Settings
config = BulkCollectionConfig()

print("📋 **Global Configuration**")
print("=" * 40)
print(f"📅 Date Range: {config.start_date} → {config.end_date}")
print(f"📊 Period: {config.default_period_days} days")
print(f"⚡ Batch Sizes: {config.batch_sizes}")
print(f"⏱️ Delays: {config.delays}")
print(f"🔄 Max Retries: {config.max_retries}")
print(f"🛡️ Parallel Workers: {config.max_workers}")

print("\n🎯 **Configuration Ready!**")
print("Choose your approach below based on your needs.")


# %% [raw]
# ---
# # 🎯 **Choose Your Approach**
#
# Select the section that matches your experience level and requirements:
#
# | 🔰 **Beginner** | 📊 **Intermediate** | 🚀 **Advanced** | ⚡ **Enterprise** |
# |-----------------|---------------------|------------------|-------------------|
# | 10-50 stocks | 50-200 stocks | 200-500 stocks | 500+ stocks |
# | Safe & Simple | Sector Analysis | Full Universe | Parallel Processing |
# | 2-5 minutes | 10-30 minutes | 30-120 minutes | 2+ hours |
#

# %% [raw]
# ## 🔰 **Level 1: Beginner - Quick Start**
#
# **Perfect for**: Learning, testing, small analysis projects
# **Time Required**: 2-5 minutes
# **Stocks**: 10-50 stocks
#
# ### Features:
# - ✅ Conservative rate limiting (safe for any API)
# - ✅ Small batch sizes (easy to manage)
# - ✅ Clear progress tracking
# - ✅ Automatic error handling
#

# %%
# Beginner Approach: Quick & Safe Stock Universe Exploration
print("🔰 **BEGINNER LEVEL: Stock Universe Overview**")
print("=" * 60)

# Explore available stock categories (no API calls)
print("\n📊 **Available HK Stock Sectors:**")
sector_summary = []
for sector, stocks in MAJOR_HK_STOCKS.items():
    sector_summary.append({
        'Sector': sector.replace('_', ' ').title(),
        'Count': len(stocks),
        'Examples': ', '.join(stocks[:3])
    })

sector_df = pd.DataFrame(sector_summary)
display(sector_df)

# Get major stocks overview
all_major_stocks = get_hk_stock_list_static()
print(f"\n📈 **Total Major Stocks Available**: {len(all_major_stocks)}")
print(f"🔍 **Sample Tickers**: {all_major_stocks[:10]}")

# Select specific sectors for analysis
tech_stocks = get_hk_stocks_by_sector('tech_stocks')
finance_stocks = get_hk_stocks_by_sector('finance')

print(f"\n💡 **Sector Breakdown:**")
print(f"   💻 Tech Sector: {len(tech_stocks)} stocks")
print(f"   🏦 Finance Sector: {len(finance_stocks)} stocks")
print(f"   📊 Total Unique: {len(set(tech_stocks + finance_stocks))} stocks")

print("\n✅ **Stock Universe Exploration Complete!**")
print("📋 Ready to fetch data using the improved utilities below.")


# %%
# Beginner Demo: Fetch Top 10 Stocks with Enhanced Utilities
print("🚀 **BEGINNER DEMO: Fetching Top 10 HK Stocks**")
print("Using improved utilities for better experience!")

# Create beginner-optimized collector
beginner_collector = create_beginner_collector()

print(f"\n📋 **Beginner Configuration:**")
print(f"   ⏱️ Delay: {beginner_collector.config.get_delay('normal')}s (conservative)")
print(f"   📦 Batch Size: {beginner_collector.config.get_batch_size('medium')} stocks")
print(f"   🔄 Max Retries: {beginner_collector.config.max_retries}")

# Select first 10 major stocks for demo
demo_stocks = all_major_stocks[:10]
print(f"\n🎯 **Target Stocks**: {demo_stocks}")

# Fetch data using improved collector
print(f"\n🔄 **Starting Fetch Process...**")
results = beginner_collector.fetch_sequential(
    stock_list=demo_stocks,
    fetch_function=fetch_hk_stocks_bulk,
    level='small'
)

# Display results using improved manager
print(f"\n📊 **RESULTS SUMMARY**")
ResultsManager.display_summary(results['data'], "Beginner Demo Results")

# Show performance statistics
stats = results['statistics']
print(f"\n⚡ **Performance Metrics:**")
print(f"   ⏱️ Total Time: {stats['total_time']:.1f}s")
print(f"   📈 Success Rate: {stats['success_rate']:.1%}")
print(f"   🚀 Processing Rate: {stats['processing_rate']:.2f} stocks/sec")
print(f"   ✅ Successful: {stats['successful']} stocks")
print(f"   ❌ Failed: {stats['failed']} stocks")

# Store results for potential use in other cells
beginner_results = results
print(f"\n🎉 **Beginner Demo Complete!** Results stored in 'beginner_results'.")


# %% [raw]
# ## 📊 **Level 2: Intermediate - Sector Analysis**
#
# **Perfect for**: Targeted analysis, sector research, comparative studies
# **Time Required**: 10-30 minutes  
# **Stocks**: 50-200 stocks
#
# ### Features:
# - ✅ Sector-specific data collection
# - ✅ Optimized batch processing
# - ✅ Comprehensive progress tracking
# - ✅ Advanced error recovery
#

# %%
# Intermediate: Sector-Focused Analysis
print("📊 **INTERMEDIATE LEVEL: Sector-Based Collection**")
print("=" * 60)

# Create standard collector for intermediate use
intermediate_collector = BulkCollector(config)

# Select sectors for analysis (you can modify this)
selected_sectors = ['tech_stocks', 'finance']  # Modify as needed
print(f"🎯 **Selected Sectors**: {[s.replace('_', ' ').title() for s in selected_sectors]}")

# Collect data for each sector
sector_results = {}
total_stocks_processed = 0

for sector in selected_sectors:
    print(f"\n🔄 **Processing {sector.replace('_', ' ').title()} Sector**")
    
    if sector == 'tech_stocks':
        # Use specialized tech fetcher
        print("   💻 Using specialized tech stock fetcher...")
        sector_data = fetch_hk_tech_stocks(
            start_date=config.start_date,
            end_date=config.end_date,
            batch_size=config.get_batch_size('medium'),
            delay_between_batches=config.get_delay('normal')
        )
        # Convert to our result format
        sector_result = {
            'data': sector_data,
            'statistics': {
                'successful': len(sector_data),
                'failed': 0,
                'total_time': 0,  # Would be calculated in real fetch
                'success_rate': 1.0 if sector_data else 0.0
            }
        }
    else:
        # Use generic bulk fetcher for other sectors
        sector_stocks = get_hk_stocks_by_sector(sector)
        print(f"   📊 Fetching {len(sector_stocks)} stocks...")
        
        sector_result = intermediate_collector.fetch_sequential(
            stock_list=sector_stocks,
            fetch_function=fetch_hk_stocks_bulk,
            level='medium'
        )
    
    sector_results[sector] = sector_result
    stocks_fetched = len(sector_result['data'])
    total_stocks_processed += stocks_fetched
    
    print(f"   ✅ {sector.replace('_', ' ').title()}: {stocks_fetched} stocks fetched")
    
    # Display sector summary
    ResultsManager.display_summary(
        sector_result['data'], 
        f"{sector.replace('_', ' ').title()} Sector Results"
    )

# Overall summary
print(f"\n🎉 **INTERMEDIATE ANALYSIS COMPLETE**")
print(f"📊 **Total Stocks Processed**: {total_stocks_processed}")
print(f"🏢 **Sectors Analyzed**: {len(selected_sectors)}")

# Show comparative statistics
print(f"\n📈 **Sector Comparison:**")
for sector, result in sector_results.items():
    stats = result['statistics']
    print(f"   {sector.replace('_', ' ').title()}: {stats['successful']} stocks "
          f"({stats['success_rate']:.1%} success rate)")

# Store results for potential use
intermediate_results = sector_results
print(f"\n💾 Results stored in 'intermediate_results' for further analysis.")


# %% [raw]
# ## 🚀 **Level 3: Advanced - Full Universe Discovery**
#
# **Perfect for**: Comprehensive research, market analysis, data science projects
# **Time Required**: 30-120 minutes
# **Stocks**: 200-500+ stocks
#
# ### Features:
# - ✅ Complete HK stock universe discovery
# - ✅ Systematic sampling strategies
# - ✅ Checkpoint support for resume capability
# - ✅ Memory-efficient processing
# - ✅ Comprehensive error handling
#

# %%
# Advanced: Full Universe Discovery and Analysis
print("🚀 **ADVANCED LEVEL: Complete HK Universe Discovery**")
print("=" * 70)

# Discover complete HK stock universe
print("🔍 **Discovering Complete HK Stock Universe...**")
print("⚠️ This may take a few minutes for validation...")

universe_config = {
    'include_major': True,
    'validate_tickers': True,
    'max_tickers': 200  # Adjust: None for complete universe, 200 for demo
}

stock_universe = get_comprehensive_hk_stock_list(**universe_config)
all_discovered_stocks = sorted(stock_universe['valid_stocks'])

print(f"\n📊 **Universe Discovery Results:**")
print(f"   ✅ Valid Stocks: {len(stock_universe['valid_stocks'])}")
print(f"   ❌ Invalid Stocks: {len(stock_universe['invalid_stocks'])}")
print(f"   📈 Range: {all_discovered_stocks[0]} → {all_discovered_stocks[-1]}")

# Resource estimation for full universe
total_stocks = len(all_discovered_stocks)
estimated_time_hours = total_stocks * 1.5 / 3600
estimated_api_calls = total_stocks
estimated_data_mb = total_stocks * 0.1

print(f"\n⚖️ **Resource Requirements:**")
print(f"   ⏱️ Estimated Time: {estimated_time_hours:.1f} hours")
print(f"   🌐 API Calls: {estimated_api_calls:,}")
print(f"   💾 Data Size: ~{estimated_data_mb:.1f} MB")

# Smart sampling for demonstration
DEMO_SIZE = 50  # Adjust based on your needs
if total_stocks > DEMO_SIZE:
    print(f"\n📋 **Demo Mode**: Using systematic sample of {DEMO_SIZE} stocks")
    # Systematic sampling for variety
    step = max(1, total_stocks // DEMO_SIZE)
    demo_stocks = all_discovered_stocks[::step][:DEMO_SIZE]
    print(f"   📍 Sample: Every {step}th stock")
else:
    demo_stocks = all_discovered_stocks
    print(f"\n📋 **Complete Set**: Using all {total_stocks} discovered stocks")

print(f"🎯 **Target for Advanced Demo**: {len(demo_stocks)} stocks")
print(f"📊 **Sample Range**: {demo_stocks[0]} → {demo_stocks[-1]}")

# Store universe data for next cell
advanced_stock_universe = demo_stocks
print(f"\n✅ **Universe Discovery Complete!** Ready for advanced fetching.")


# %%
# Advanced: Execute with Checkpointing Support
print("🚧 **ADVANCED EXECUTION: Fetch with Checkpoint Support**")
print("=" * 60)

# Create advanced collector
advanced_collector = BulkCollector(config)

# Safety check and user confirmation
EXECUTE_ADVANCED = True  # Set to True to execute
CHECKPOINT_DIR = "data/advanced_checkpoints"

print(f"⚙️ **Advanced Configuration:**")
print(f"   📦 Batch Size: {config.get_batch_size('large')}")
print(f"   ⏱️ Delay: {config.get_delay('conservative')}s")
print(f"   💾 Checkpoint Dir: {CHECKPOINT_DIR}")
print(f"   🎯 Target Stocks: {len(advanced_stock_universe)}")

if not EXECUTE_ADVANCED:
    print("\n🛡️ **Safety Mode**: Advanced execution disabled")
    print("💡 Set EXECUTE_ADVANCED = True to run comprehensive fetch")
    print(f"📊 Ready to process {len(advanced_stock_universe)} stocks")
    
else:
    print("\n🚀 **EXECUTING ADVANCED FETCH WITH CHECKPOINTS**")
    
    # Option 1: Checkpoint-enabled fetch (recommended for large operations)
    print("🔄 **Method: Checkpoint-Enabled Fetch**")
    print("💡 Can be interrupted and resumed at any time")
    
    advanced_results = advanced_collector.fetch_with_checkpoints(
        stock_list=advanced_stock_universe,
        fetch_function=fetch_hk_stocks_bulk,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_every=25  # Save progress every 25 stocks
    )
    
    # Display comprehensive results
    if advanced_results['data']:
        print(f"\n📊 **ADVANCED RESULTS SUMMARY**")
        ResultsManager.display_summary(
            advanced_results['data'], 
            "Advanced Universe Discovery Results"
        )
        
        print(f"\n🎉 **Advanced Analysis Complete!**")
        print(f"✅ Successfully fetched: {len(advanced_results['data'])} stocks")
        print(f"❌ Failed: {len(advanced_results['failed'])} stocks")
        
        # Save comprehensive dataset
        print(f"\n💾 **Saving Advanced Dataset...**")
        try:
            save_result = ResultsManager.save_with_metadata(
                data_dict=advanced_results['data'],
                metadata={
                    'collection_type': 'advanced_universe',
                    'total_stocks': len(advanced_stock_universe),
                    'successful': len(advanced_results['data']),
                    'failed': len(advanced_results['failed']),
                    'timestamp': datetime.now().isoformat(),
                    'config': config.__dict__
                },
                output_dir="data/advanced_universe"
            )
            print(f"✅ Dataset saved: {save_result['total_files']} files")
            print(f"📄 Metadata: {save_result['metadata_file']}")
            
        except Exception as e:
            print(f"⚠️ Save operation failed: {e}")
    
    else:
        print("❌ No data collected - check error logs")

# Store results
if 'advanced_results' in locals():
    print(f"\n💾 Results stored in 'advanced_results' variable")
else:
    print(f"\n💡 Execute the cell with EXECUTE_ADVANCED = True to run the fetch")


# %% [raw]
# ## ⚡ **Level 4: Enterprise - Parallel Processing**
#
# **Perfect for**: Production systems, time-critical analysis, large-scale operations
# **Time Required**: 2+ hours (depending on scale)
# **Stocks**: 500+ stocks
#
# ### Features:
# - ✅ Safe parallel processing with rate limiting
# - ✅ Enterprise-grade error handling
# - ✅ Performance monitoring and optimization
# - ✅ Production deployment templates
#
# ### ⚠️ **WARNING**: Use with extreme caution - can overwhelm APIs!
#

# %%
# Enterprise: Parallel Processing Demo with Safety Controls
print("⚡ **ENTERPRISE LEVEL: Parallel Processing with Safety**")
print("=" * 70)

# Enterprise configuration
class EnterpriseConfig:
    """Enterprise-specific settings with safety controls"""
    MAX_WORKERS = 2  # Conservative - increase after testing
    WORKER_DELAY = 1.0  # Delay per worker request
    ENABLE_PARALLEL = False  # Safety switch - must be explicitly enabled
    DEMO_SIZE = 10  # Small demo for safety

print(f"🛡️ **Enterprise Safety Configuration:**")
print(f"   ⚡ Max Workers: {EnterpriseConfig.MAX_WORKERS}")
print(f"   ⏱️ Worker Delay: {EnterpriseConfig.WORKER_DELAY}s")
print(f"   🔒 Parallel Enabled: {EnterpriseConfig.ENABLE_PARALLEL}")
print(f"   📊 Demo Size: {EnterpriseConfig.DEMO_SIZE}")

# Create enterprise collector
enterprise_collector = create_enterprise_collector()

# Demo dataset for parallel processing
if 'all_major_stocks' in locals():
    demo_parallel_stocks = all_major_stocks[:EnterpriseConfig.DEMO_SIZE]
else:
    # Fallback demo stocks
    demo_parallel_stocks = ['0700.HK', '0005.HK', '0941.HK', '1299.HK', '2318.HK']

print(f"\n🎯 **Parallel Demo Target**: {demo_parallel_stocks}")

if EnterpriseConfig.ENABLE_PARALLEL:
    print(f"\n⚡ **EXECUTING PARALLEL PROCESSING**")
    print(f"🚨 WARNING: Using {EnterpriseConfig.MAX_WORKERS} workers with {EnterpriseConfig.WORKER_DELAY}s delays")
    
    # Execute parallel processing
    parallel_results = enterprise_collector.fetch_parallel(
        stock_list=demo_parallel_stocks,
        fetch_function=fetch_hk_stocks_bulk,
        max_workers=EnterpriseConfig.MAX_WORKERS
    )
    
    # Display results
    print(f"\n📊 **PARALLEL PROCESSING RESULTS**")
    ResultsManager.display_summary(
        parallel_results['data'],
        "Enterprise Parallel Processing Results"
    )
    
    # Performance analysis
    stats = parallel_results['statistics']
    print(f"\n⚡ **Performance Analysis:**")
    print(f"   ⏱️ Total Time: {stats['total_time']:.1f}s")
    print(f"   📈 Success Rate: {stats['success_rate']:.1%}")
    print(f"   🚀 Processing Rate: {stats['processing_rate']:.2f} stocks/sec")
    print(f"   👥 Workers Used: {parallel_results['metadata']['max_workers']}")
    
    # Compare with sequential baseline
    print(f"\n📊 **Sequential vs Parallel Comparison:**")
    print(f"   Sequential (estimated): {len(demo_parallel_stocks) * EnterpriseConfig.WORKER_DELAY:.1f}s")
    print(f"   Parallel (actual): {stats['total_time']:.1f}s")
    speedup = (len(demo_parallel_stocks) * EnterpriseConfig.WORKER_DELAY) / stats['total_time']
    print(f"   🚀 Speedup: {speedup:.1f}x faster")
    
    enterprise_results = parallel_results
    
else:
    print(f"\n🛡️ **PARALLEL PROCESSING DISABLED FOR SAFETY**")
    print(f"🔧 **To Enable Enterprise Parallel Processing:**")
    print(f"   1. Set EnterpriseConfig.ENABLE_PARALLEL = True")
    print(f"   2. Test with small datasets first (5-10 stocks)")
    print(f"   3. Monitor API response times carefully")
    print(f"   4. Gradually increase workers (2 → 4 → 8)")
    print(f"   5. Implement comprehensive monitoring")
    print(f"   6. Have fallback to sequential processing")
    
    print(f"\n⚡ **Enterprise Infrastructure Ready:**")
    print(f"   📊 Demo Stocks: {len(demo_parallel_stocks)}")
    print(f"   🔧 Workers Available: {EnterpriseConfig.MAX_WORKERS}")
    estimated_time = len(demo_parallel_stocks) * EnterpriseConfig.WORKER_DELAY / EnterpriseConfig.MAX_WORKERS
    print(f"   ⏱️ Estimated Time: {estimated_time:.1f}s")
    print(f"   💡 Ready for production deployment!")

print(f"\n✅ **Enterprise Level Complete**")


# %% [raw]
# ---
# ## 📊 **Summary & Best Practices**
#
# ### 🎯 **Quick Reference Guide**
#
# | Your Need | Level | Typical Time | Best Approach |
# |-----------|-------|--------------|---------------|
# | Learning & Testing | 🔰 Beginner | 2-5 min | `create_beginner_collector()` |
# | Sector Analysis | 📊 Intermediate | 10-30 min | `BulkCollector()` with sector lists |
# | Market Research | 🚀 Advanced | 30-120 min | `fetch_with_checkpoints()` |
# | Production System | ⚡ Enterprise | 2+ hours | `create_enterprise_collector()` |
#
# ### 🛡️ **Critical Success Factors**
# 1. **Start Small**: Always test with 5-10 stocks first
# 2. **Rate Limiting**: Respect API constraints (1-2s delays minimum)  
# 3. **Error Handling**: Implement retry logic and checkpointing
# 4. **Monitoring**: Track success rates and performance metrics
# 5. **Progressive Scaling**: Gradually increase batch sizes and workers
#
# ### 📈 **Performance Optimization Tips**
# - Use checkpointing for operations > 100 stocks
# - Monitor memory usage for large datasets
# - Implement caching for frequently accessed data  
# - Use systematic sampling for universe discovery
# - Have fallback strategies for API failures
#

# %%
# Final Summary and Next Steps
print("🎉 **BULK DATA COLLECTION NOTEBOOK COMPLETE!**")
print("=" * 70)

# Collect all results if available
all_results = {}
result_summary = []

# Check what results we have from each level
if 'beginner_results' in locals():
    all_results['beginner'] = beginner_results
    result_summary.append({
        'Level': '🔰 Beginner',
        'Stocks': len(beginner_results['data']),
        'Success_Rate': f"{beginner_results['statistics']['success_rate']:.1%}",
        'Time': f"{beginner_results['statistics']['total_time']:.1f}s"
    })

if 'intermediate_results' in locals():
    total_intermediate = sum(len(r['data']) for r in intermediate_results.values())
    all_results['intermediate'] = intermediate_results
    result_summary.append({
        'Level': '📊 Intermediate',
        'Stocks': total_intermediate,
        'Success_Rate': 'Varies by sector',
        'Time': 'Varies by sector'
    })

if 'advanced_results' in locals():
    all_results['advanced'] = advanced_results
    result_summary.append({
        'Level': '🚀 Advanced',
        'Stocks': len(advanced_results['data']) if 'data' in advanced_results else 0,
        'Success_Rate': 'Checkpointed',
        'Time': 'Variable'
    })

if 'enterprise_results' in locals():
    all_results['enterprise'] = enterprise_results
    result_summary.append({
        'Level': '⚡ Enterprise',
        'Stocks': len(enterprise_results['data']),
        'Success_Rate': f"{enterprise_results['statistics']['success_rate']:.1%}",
        'Time': f"{enterprise_results['statistics']['total_time']:.1f}s"
    })

# Display session summary
if result_summary:
    print(f"\n📊 **SESSION RESULTS SUMMARY**")
    summary_df = pd.DataFrame(result_summary)
    display(summary_df)
    
    total_stocks = sum(int(str(row['Stocks']).split()[0]) if isinstance(row['Stocks'], str) else row['Stocks'] 
                      for row in result_summary)
    print(f"\n🎯 **Total Stocks Processed This Session**: {total_stocks}")
else:
    print(f"\n💡 **No results collected** - Run the level sections above to see data collection in action!")

print(f"\n🚀 **WHAT YOU'VE ACCOMPLISHED:**")
print(f"✅ Learned 4 different approaches to bulk data collection")
print(f"✅ Used production-ready utilities with error handling")
print(f"✅ Experienced progressive complexity (Beginner → Enterprise)")
print(f"✅ Gained tools for any scale of HK stock analysis")

print(f"\n📋 **NEXT STEPS:**")
print(f"1. 🔧 **Customize**: Modify configurations for your specific needs")
print(f"2. 📊 **Scale**: Apply to your target stock universe size")
print(f"3. 🏭 **Deploy**: Use enterprise features for production systems")
print(f"4. 📈 **Analyze**: Process your collected data with feature extraction")
print(f"5. 🔄 **Iterate**: Refine and optimize based on your results")

print(f"\n💡 **AVAILABLE UTILITIES:**")
print(f"   📦 BulkCollectionConfig - Centralized configuration")
print(f"   🔧 BulkCollector - Main collection engine")  
print(f"   📊 ResultsManager - Data display and saving")
print(f"   🔰 create_beginner_collector() - Safe defaults")
print(f"   ⚡ create_enterprise_collector() - High performance")

print(f"\n" + "="*70)
print(f"✅ **SUCCESS!** You're now equipped for professional-grade HK stock data collection! 🇭🇰📈")

# Save all configuration for reference
final_config = {
    'session_timestamp': datetime.now().isoformat(),
    'levels_completed': list(all_results.keys()),
    'total_stocks_processed': total_stocks if 'total_stocks' in locals() else 0,
    'configuration_used': config.__dict__
}

print(f"\n💾 Session summary available in 'final_config' variable")

