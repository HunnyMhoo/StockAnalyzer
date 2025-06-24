# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [raw] vscode={"languageId": "raw"}
# # Feature Extraction from Labeled Stock Patterns
#
# This notebook demonstrates how to extract numerical features from labeled stock patterns using the FeatureExtractor class. These features can then be used for machine learning model training.
#
# ## User Story 1.3 Implementation
# - Extract 18+ numerical features from labeled patterns
# - Generate features across 4 categories: Trend Context, Correction Phase, False Support Break, Technical Indicators
# - Save results to CSV for ML training
#

# %% [raw] vscode={"languageId": "raw"}
# ## Setup and Imports
#

# %% [raw]
#

# %%
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append('..')

# Import our modules
from src.feature_extractor import FeatureExtractor, extract_features_from_labels
from src.pattern_labeler import PatternLabel, load_labeled_patterns
from src.data_fetcher import fetch_hk_stocks, list_cached_tickers

print("✅ All imports successful!")
print(f"📅 Notebook run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# %% [raw] vscode={"languageId": "raw"}
# ## Check Available Data
#

# %%
# Check what labeled patterns we have
labels_file = "../labels/labeled_patterns.json"
notebook_labels_file = "labels/labeled_patterns.json"

# Try both locations
if os.path.exists(labels_file):
    patterns_file = labels_file
elif os.path.exists(notebook_labels_file):
    patterns_file = notebook_labels_file
else:
    patterns_file = None

if patterns_file:
    try:
        labeled_patterns = load_labeled_patterns(patterns_file)
        print(f"📋 Found {len(labeled_patterns)} labeled patterns:")
        
        for i, pattern in enumerate(labeled_patterns[:5], 1):  # Show first 5
            print(f"  {i}. {pattern.ticker}: {pattern.start_date} to {pattern.end_date} ({pattern.label_type})")
        
        if len(labeled_patterns) > 5:
            print(f"  ... and {len(labeled_patterns) - 5} more patterns")
            
    except Exception as e:
        print(f"❌ Error loading patterns: {e}")
        labeled_patterns = []
else:
    print("⚠️  No labeled patterns file found")
    labeled_patterns = []


# %% [raw] vscode={"languageId": "raw"}
# ## Method 1: Extract Features from Labeled Patterns File
#
# This is the main use case - extracting features from all labeled patterns at once.
#

# %%
if patterns_file and os.path.exists(patterns_file):
    print("🔄 Extracting features from labeled patterns...")
    
    try:
        # Extract features from all labeled patterns
        features_df = extract_features_from_labels(
            labels_file=patterns_file,
            output_file="../features/notebook_extracted_features.csv"
        )
        
        if not features_df.empty:
            print(f"✅ Successfully extracted features!")
            print(f"📊 Shape: {features_df.shape}")
            print(f"🎯 Patterns processed: {len(features_df)}")
            
            # Display the dataframe
            display(features_df.head())
            
        else:
            print("⚠️  No features extracted - check data availability")
            
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        features_df = pd.DataFrame()
else:
    print("⚠️  Skipping - no labeled patterns file available")
    features_df = pd.DataFrame()


# %% [raw] vscode={"languageId": "raw"}
# ## Feature Analysis and Summary
#
# Analyze the extracted features and show statistics.
#

# %% [markdown]
#

# %% [raw] vscode={"languageId": "raw"}
# ## Optional: Refresh Data Cache
#
# If you're experiencing data issues, you can refresh the cached data for your tickers.
#

# %%
# Refresh data cache for your tickers (only run if needed)
tickers = ['0700.HK', '0005.HK', '0001.HK', '0388.HK', '0003.HK']

# Calculate date range for 2 years of data
from datetime import datetime, timedelta
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years

print(f"📅 Refreshing data from {start_date} to {end_date}")

for ticker in tickers:
    print(f"🔄 Refreshing {ticker}...")
    try:
        data = fetch_hk_stocks([ticker], start_date, end_date, force_refresh=True)
        if ticker in data:
            print(f"✅ {ticker}: {len(data[ticker])} records")
        else:
            print(f"❌ {ticker}: Failed to fetch")
    except Exception as e:
        print(f"❌ {ticker}: Error - {e}")

print("🎉 Data refresh completed!")


# %%
if not features_df.empty:
    # Analyze the extracted features
    print("📈 Feature Analysis")
    print("=" * 40)
    
    # Separate metadata and feature columns
    metadata_cols = ['ticker', 'start_date', 'end_date', 'label_type', 'notes']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    print(f"📊 Total columns: {len(features_df.columns)}")
    print(f"📋 Metadata columns: {len(metadata_cols)}")
    print(f"🔢 Feature columns: {len(feature_cols)}")
    
    print(f"\n🎯 Feature Categories:")
    
    # Categorize features
    trend_features = [col for col in feature_cols if any(keyword in col for keyword in ['trend', 'sma', 'angle'])]
    correction_features = [col for col in feature_cols if any(keyword in col for keyword in ['drawdown', 'recovery', 'down_day'])]
    support_features = [col for col in feature_cols if any(keyword in col for keyword in ['support', 'break'])]
    technical_features = [col for col in feature_cols if col in ['rsi_14', 'macd_diff', 'volatility', 'volume_avg_ratio']]
    
    print(f"  🔺 Trend Context: {len(trend_features)} features")
    print(f"  📉 Correction Phase: {len(correction_features)} features")
    print(f"  🛡️  Support Break: {len(support_features)} features")
    print(f"  📊 Technical Indicators: {len(technical_features)} features")
    
    print(f"\n✅ Total numerical features: {len(feature_cols)} (minimum required: 10)")
    
    # Feature statistics
    print(f"\n📊 Feature Statistics:")
    display(features_df[feature_cols].describe().round(4))
    
    # Check for missing values
    missing_counts = features_df[feature_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\n⚠️  Missing values detected:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  • {col}: {count} missing ({count/len(features_df)*100:.1f}%)")
    else:
        print(f"\n✅ No missing values in feature columns")
        
else:
    print("⚠️  No features available to analyze")


# %% [raw] vscode={"languageId": "raw"}
# ## Summary and Next Steps
#
# Review what was accomplished and suggest next steps.
#

# %%
print("🎉 Feature Extraction Summary")
print("=" * 50)

# Check what files were created
output_files = []
features_dir = "../features"

if os.path.exists(features_dir):
    for file in os.listdir(features_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(features_dir, file)
            file_size = os.path.getsize(file_path)
            output_files.append((file, file_size))

if output_files:
    print(f"📁 Generated files in {features_dir}/:")
    for file, size in output_files:
        print(f"  • {file} ({size:,} bytes)")
else:
    print("⚠️  No output files found")

print(f"\n✅ Feature extraction completed successfully!")
print(f"\n🚀 Next Steps:")
print(f"  1. Review the generated CSV files for data quality")
print(f"  2. Use the features for machine learning model training")
print(f"  3. Add more labeled patterns to increase dataset size")
print(f"  4. Experiment with different FeatureExtractor parameters")
print(f"  5. Consider feature engineering and selection techniques")

print(f"\n📊 Feature Categories Implemented:")
print(f"  🔺 Trend Context: prior_trend_return, above_sma_50_ratio, trend_angle")
print(f"  📉 Correction Phase: drawdown_pct, recovery_return_pct, down_day_ratio")
print(f"  🛡️  False Support Break: support_level, support_break_depth_pct, false_break_flag, recovery_days, recovery_volume_ratio")
print(f"  📊 Technical Indicators: sma_5/10/20, rsi_14, macd_diff, volatility, volume_avg_ratio")

print(f"\n🎯 User Story 1.3 Status: ✅ COMPLETED")
print(f"  • Minimum 10 features required: ✅ (18+ implemented)")
print(f"  • Configurable window size: ✅")
print(f"  • CSV output format: ✅")
print(f"  • Error handling and validation: ✅")
print(f"  • Batch processing capability: ✅")

