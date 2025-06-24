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

# %% [markdown] magic_args="[markdown]"
# # 🎯 Advanced Hong Kong Stock Pattern Finder
# This notebook finds stocks with patterns similar to a user-defined example, using user-provided negative examples for more accurate training.
#
# ## Workflow:
# 1.  **Define Positive Pattern** → Enter one stock ticker and date range that represents the pattern you want to find.
# 2.  **Define Negative Examples** → Provide a comma-separated list of stock tickers that explicitly **do not** show the desired pattern.
# 3.  **Find Matches** → The system trains a temporary model on your examples and scans the market for similar patterns.

# %%
# %%
# CELL 1: SETUP - Imports and Path Configuration
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# %%
# Setup paths
current_dir = Path.cwd()
project_root = current_dir.parent if current_dir.name == 'notebooks' else current_dir
sys.path.insert(0, str(project_root))

# %%
# Import custom modules
from src.data_fetcher import fetch_hk_stocks, get_all_cached_tickers, validate_cached_data_file
from src.feature_extractor import FeatureExtractor
from src.pattern_scanner import PatternScanner, ScanningConfig
from src.hk_stock_universe import get_hk_stock_list_static

# %%
warnings.filterwarnings('ignore')
print("✅ Setup Complete: All libraries and modules are loaded.")

# %% [markdown]
# ## 📊 Data Discovery
# 
# Let's check what stock data is available for pattern scanning:

# %%
def show_data_summary():
    """Display summary of available stock data for pattern scanning"""
    print("🔍 Discovering available stock data...")
    available_stocks = get_all_cached_tickers()
    
    if not available_stocks:
        print("❌ No cached stock data found!")
        print("📝 Please run data collection notebooks first (01_data_collection.py or 02_bulk_data_collection.py)")
        return
    
    print(f"✅ Found {len(available_stocks)} stocks with cached data")
    print(f"📈 Sample stocks: {', '.join(available_stocks[:10])}")
    if len(available_stocks) > 10:
        print(f"   ... and {len(available_stocks) - 10} more")
    
    # Show data quality overview
    print("\n📊 Data Quality Overview:")
    high_quality = medium_quality = low_quality = 0
    
    for ticker in available_stocks[:20]:  # Sample first 20 for performance
        validation = validate_cached_data_file(ticker)
        if validation['data_quality_score'] >= 0.8:
            high_quality += 1
        elif validation['data_quality_score'] >= 0.6:
            medium_quality += 1
        else:
            low_quality += 1
    
    total_sampled = min(20, len(available_stocks))
    print(f"   High Quality: {high_quality}/{total_sampled} stocks")
    print(f"   Medium Quality: {medium_quality}/{total_sampled} stocks") 
    print(f"   Low Quality: {low_quality}/{total_sampled} stocks")
    
    print(f"\n🎯 Ready to scan {len(available_stocks)} stocks for patterns!")

show_data_summary()


# %%
# %%
# Simple config class that can be pickled (must be at module level)
class SimpleConfig:
    def __init__(self):
        self.model_type = "xgboost"


# %%
# %%
# CELL 2: PATTERN ANALYSIS FUNCTION (FIXED VERSION)
def find_similar_patterns(positive_ticker, start_date_str, end_date_str, negative_tickers_str):
    """
    Analyzes a given stock pattern and finds similar patterns in other stocks, using user-defined negative examples.
    """
    try:
        # --- 1. Validate Inputs ---
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        if not positive_ticker.endswith('.HK') or (end_date - start_date).days < 5:
            raise ValueError("Invalid positive pattern. Ticker must be a .HK stock and pattern must be >= 5 days.")

        # Parse and validate negative tickers
        negative_tickers = [t.strip().upper() for t in negative_tickers_str.split(',') if t.strip()]
        if not negative_tickers:
            raise ValueError("Please provide at least one negative ticker.")
        for t in negative_tickers:
            if not t.endswith('.HK'):
                raise ValueError(f"Invalid negative ticker: {t}. All tickers must end with .HK")

        print(f"🔍 Analyzing positive pattern for {positive_ticker} from {start_date_str} to {end_date_str}...")
        print(f"📉 Using {len(negative_tickers)} negative examples: {', '.join(negative_tickers)}")

        # --- 2. Extract Features for the Positive Pattern ---
        extractor = FeatureExtractor()
        context_start_date = start_date - timedelta(days=30)
        data_dict = fetch_hk_stocks([positive_ticker], str(context_start_date), str(end_date))
        if not data_dict or positive_ticker not in data_dict:
            raise ConnectionError(f"Could not fetch data for positive ticker {positive_ticker}.")

        full_data = data_dict[positive_ticker]
        window_data = full_data.loc[start_date_str:end_date_str]
        prior_context_data = full_data.loc[:start_date_str].iloc[:-1]

        positive_features = extractor.extract_features_from_window_data(
            window_data, prior_context_data, positive_ticker, start_date_str, end_date_str, full_data
        )
        if not positive_features:
            raise ValueError("Could not extract features from the provided positive pattern.")

        # --- 3. Extract Features for Negative Examples ---
        all_features = [positive_features]
        all_labels = [1]
        
        negative_data = fetch_hk_stocks(negative_tickers, (end_date - timedelta(days=365)).strftime('%Y-%m-%d'), str(end_date))

        for neg_ticker, neg_df in negative_data.items():
            if len(neg_df) > len(window_data) + 30:
                rand_start = np.random.randint(0, len(neg_df) - len(window_data) - 30)
                neg_window = neg_df.iloc[rand_start + 30 : rand_start + 30 + len(window_data)]
                neg_context = neg_df.iloc[rand_start : rand_start + 30]
                
                neg_features = extractor.extract_features_from_window_data(
                    neg_window, neg_context, neg_ticker, str(neg_window.index.min().date()), str(neg_window.index.max().date()), neg_df
                )
                if neg_features:
                    all_features.append(neg_features)
                    all_labels.append(0)
        
        if all_labels.count(0) == 0:
            raise ValueError("Failed to generate negative training samples from the provided tickers.")
            
        # Create a DataFrame from all collected features (positive and negative)
        full_features_df = pd.DataFrame(all_features)

        # Define all metadata columns that should NOT be used for training
        metadata_cols = ['ticker', 'start_date', 'end_date', 'label_type', 'notes']
        
        # Get the final list of feature names by excluding metadata
        feature_names = [col for col in full_features_df.columns if col not in metadata_cols]
        
        # Create a clean DataFrame for training with only numeric features
        training_df_raw = full_features_df[feature_names]

        # Force all training columns to be numeric, coercing errors to NaN
        training_df = training_df_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # --- 4. Train a Temporary Model ---
        print(f"🤖 Training model on {len(training_df)} samples ({all_labels.count(1)} positive, {all_labels.count(0)} negative)...")
        
        try:
            import xgboost as xgb
            import joblib
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(training_df, pd.Series(all_labels))
        except Exception as e:
            raise Exception(f"Model training failed: {e}")

        # --- 5. Scan for Similar Patterns ---
        print(f"🔎 Discovering available stocks from cached data...")
        all_available_stocks = get_all_cached_tickers()
        
        if not all_available_stocks:
            print("⚠️  No cached stock data found. Please run data collection first.")
            return
            
        print(f"📊 Found {len(all_available_stocks)} stocks with cached data")
        print(f"🔎 Scanning for similar patterns...")
        
        # Create a model package compatible with PatternScanner
        # This fixes the "argument of type 'XGBClassifier' is not iterable" error
        # Using the module-level SimpleConfig class to avoid pickling issues
        model_package = {
            'model': model,
            'scaler': None,
            'feature_names': feature_names,
            'config': SimpleConfig(),
            'metadata': {
                'training_date': datetime.now().isoformat(),
                'n_samples': len(training_df),
                'n_features': len(feature_names),
                'class_distribution': pd.Series(all_labels).value_counts().to_dict()
            }
        }
        
        # Save the model package (not just the raw model)
        temp_model_path = "temp_model.joblib"
        joblib.dump(model_package, temp_model_path)
        
        try:
            # Initialize the scanner with the properly formatted model package
            scanner = PatternScanner(model_path=temp_model_path)
            
            scan_list = [t for t in all_available_stocks if t != positive_ticker and t not in negative_tickers]
            print(f"📊 Scanning {len(scan_list)} stocks (excluding positive and negative examples)")
            
            scan_results = scanner.scan_tickers(scan_list, ScanningConfig(min_confidence=0.7))

            if scan_results and not scan_results.matches_df.empty:
                matches_df = scan_results.matches_df.sort_values('probability', ascending=False)
                print(f"\n✅ Found {len(matches_df)} similar patterns out of {len(scan_list)} stocks scanned!")
                
                # Show confidence distribution
                high_conf = len(matches_df[matches_df['probability'] >= 0.9])
                med_conf = len(matches_df[(matches_df['probability'] >= 0.8) & (matches_df['probability'] < 0.9)])
                low_conf = len(matches_df[matches_df['probability'] < 0.8])
                
                print(f"📈 Confidence Distribution: {high_conf} high (≥90%), {med_conf} medium (80-90%), {low_conf} moderate (70-80%)")
                print(f"🎯 Top match: {matches_df.iloc[0]['ticker']} with {matches_df.iloc[0]['probability']:.1%} confidence")
                
                display(HTML(matches_df.to_html(index=False)))
            else:
                print(f"\n✅ Analysis complete. No similar patterns found out of {len(scan_list)} stocks scanned.")
                print("💡 Try lowering the confidence threshold or adjusting your pattern examples.")
        finally:
            # Clean up the temporary model file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

# %%
# %%
# CELL 3: USER INTERFACE AND EXECUTION
# Create widgets for user input
positive_ticker_input = widgets.Text(value='0700.HK', description='Positive Ticker:')
start_date_input = widgets.Text(value='2024-01-15', description='Start Date:')
end_date_input = widgets.Text(value='2024-02-05', description='End Date:')
negative_tickers_input = widgets.Text(value='0005.HK,0941.HK,0388.HK', description='Negative Tickers:')
run_button = widgets.Button(description="🔍 Find Similar Patterns", button_style='primary', layout=widgets.Layout(width='250px', height='40px'))
output_area = widgets.Output()

# %%
def on_button_click(b):
    with output_area:
        clear_output(True)
        find_similar_patterns(
            positive_ticker_input.value,
            start_date_input.value,
            end_date_input.value,
            negative_tickers_input.value
        )

# %%
run_button.on_click(on_button_click)

# %%
# Display the interface
display(
    widgets.VBox([
        widgets.HTML("<h3>Enter Pattern Details</h3>"),
        widgets.HTML("<b>Provide one positive example of the pattern you want to find.</b>"),
        positive_ticker_input,
        start_date_input,
        end_date_input,
        widgets.HTML("<hr style='margin-top: 10px; margin-bottom: 10px'>"),
        widgets.HTML("<b>Provide comma-separated negative examples (stocks that DON'T show the pattern).</b>"),
        negative_tickers_input,
        run_button,
        output_area
    ])
) 

# %%
