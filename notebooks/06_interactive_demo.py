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
# # 🎯 Advanced Hong Kong Stock Pattern Finder
#
# **Interactive pattern discovery tool** that finds stocks with patterns similar to user-defined examples.
#
# ## ✨ Features
# - **Smart Pattern Matching**: Use positive/negative examples for precise pattern detection
# - **Clean Progress Tracking**: No overwhelming log output during large scans
# - **Enhanced UI**: Improved widgets with validation and guidance
# - **Result Visualization**: Clear presentation of findings with export options
# - **Performance Optimized**: Efficient scanning with progress indicators
#
# ## 📋 Workflow
# 1. **Define Positive Pattern** → Enter stock ticker and date range showing your target pattern
# 2. **Define Negative Examples** → Provide stocks that explicitly **don't** show the pattern
# 3. **Configure Scanning** → Set confidence thresholds and scan parameters
# 4. **Find Matches** → System trains temporary model and scans market for similar patterns

# %%
# %%
# SETUP - Imports and Environment Configuration
import os
import sys
import time
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

# Use shared setup utility with smart logging
from common_setup import setup_notebook, get_hk_stock_names, import_common_modules, configure_logging

# Configure for clean output (suppress verbose logs)
configure_logging(verbose=False)

# %%
# Set up notebook environment
validation = setup_notebook(verbose_logging=False, quiet=True)

# Import custom modules
modules = import_common_modules()
fetch_hk_stocks = modules['fetch_hk_stocks']
get_all_cached_tickers = modules['get_all_cached_tickers']
FeatureExtractor = modules['FeatureExtractor']
PatternScanner = modules['PatternScanner']
get_hk_stock_list_static = modules['get_hk_stock_list_static']

# %%
# Additional imports
from src.data_fetcher import validate_cached_data_file
from src.pattern_scanner import ScanningConfig

print("✅ Setup Complete: Interactive pattern finder ready!")

# %%
# Add markdown cell header
print("## 📊 Available Data Discovery")

# %% [markdown]
# Check available stock data and quality for pattern scanning:

# %%
def show_enhanced_data_summary():
    """Enhanced data summary with quality metrics and scanning readiness"""
    print("🔍 Analyzing available stock data...")
    available_stocks = get_all_cached_tickers()
    
    if not available_stocks:
        print("❌ No cached stock data found!")
        print("📝 Please run data collection notebooks first:")
        print("   • 02_basic_data_collection.py (for beginners)")
        print("   • 02_advanced_data_collection.py (for large-scale)")
        return []
    
    print(f"✅ Found {len(available_stocks)} stocks with cached data")
    
    # Quick quality assessment
    high_quality = medium_quality = low_quality = 0
    sample_size = min(20, len(available_stocks))
    
    print(f"📊 Quality assessment (sample of {sample_size} stocks):")
    
    for ticker in available_stocks[:sample_size]:
        try:
            validation = validate_cached_data_file(ticker)
            quality_score = validation['data_quality_score']
            
            if quality_score >= 0.8:
                high_quality += 1
            elif quality_score >= 0.6:
                medium_quality += 1
            else:
                low_quality += 1
        except:
            low_quality += 1
    
    print(f"   🟢 High Quality: {high_quality}/{sample_size} stocks ({high_quality/sample_size*100:.0f}%)")
    print(f"   🟡 Medium Quality: {medium_quality}/{sample_size} stocks ({medium_quality/sample_size*100:.0f}%)")
    print(f"   🔴 Low Quality: {low_quality}/{sample_size} stocks ({low_quality/sample_size*100:.0f}%)")
    
    # Show sample of available stocks
    print(f"\n📈 Sample available stocks: {', '.join(available_stocks[:10])}")
    if len(available_stocks) > 10:
        print(f"   ... and {len(available_stocks) - 10} more")
    
    print(f"\n🎯 Ready to scan {len(available_stocks)} stocks for patterns!")
    return available_stocks

available_data = show_enhanced_data_summary()


# %%
# Simple configuration class for model compatibility
class SimplePatternConfig:
    """Lightweight config class for temporary models"""
    def __init__(self):
        self.model_type = "xgboost"
        self.training_approach = "interactive_demo"


# %%
# ENHANCED PATTERN ANALYSIS FUNCTION
def find_similar_patterns_enhanced(positive_ticker, start_date_str, end_date_str, 
                                 negative_tickers_str, min_confidence=0.7, 
                                 max_stocks_to_scan=None, show_progress=True):
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
            'config': SimplePatternConfig(),
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
            
            # Apply scan limit if specified
            if max_stocks_to_scan and max_stocks_to_scan < len(scan_list):
                scan_list = scan_list[:max_stocks_to_scan]
            
            print(f"📊 Scanning {len(scan_list)} stocks (excluding positive and negative examples)")
            
            # ISSUE 1 FIX: Suppress excessive logging during scanning
            import sys
            from io import StringIO
            
            # Capture output during scanning to reduce log spam
            old_stdout = sys.stdout
            captured_output = StringIO()
            
            progress_bar = tqdm(total=len(scan_list), desc="Scanning stocks", unit="stocks")
            
            try:
                # Temporarily redirect stdout to suppress verbose scanner output
                sys.stdout = captured_output
                
                # Scan with ZERO confidence to get ALL results (we'll filter later)
                scan_results = scanner.scan_tickers(scan_list, ScanningConfig(
                    min_confidence=0.0,  # Get all results
                    max_windows_per_ticker=3,
                    save_results=False,
                    top_matches_display=0  # Suppress internal display
                ))
                
            finally:
                sys.stdout = old_stdout
                progress_bar.close()
            
            # ISSUE 2 FIX: Show top scores even when no matches meet threshold
            if scan_results and not scan_results.matches_df.empty:
                # Get all results sorted by confidence
                all_results = scan_results.matches_df.sort_values('confidence_score', ascending=False)
                
                # Debug: Check available columns
                print(f"📊 Debug: Available columns: {list(all_results.columns)}")
                
                # Apply confidence threshold for "matches"
                matches_df = all_results[all_results['confidence_score'] >= min_confidence]
                
                if not matches_df.empty:
                    print(f"\n✅ Found {len(matches_df)} patterns meeting {min_confidence:.0%} confidence threshold!")
                    
                    # Show confidence distribution for matches
                    high_conf = len(matches_df[matches_df['confidence_score'] >= 0.9])
                    med_conf = len(matches_df[(matches_df['confidence_score'] >= 0.8) & (matches_df['confidence_score'] < 0.9)])
                    low_conf = len(matches_df[matches_df['confidence_score'] < 0.8])
                    
                    print(f"📈 Confidence Distribution: {high_conf} high (≥90%), {med_conf} medium (80-90%), {low_conf} moderate (70-80%)")
                    print(f"🎯 Top match: {matches_df.iloc[0]['ticker']} with {matches_df.iloc[0]['confidence_score']:.1%} confidence")
                    
                    # Display results table - use available columns
                    available_cols = ['ticker', 'confidence_score']
                    if 'window_start_date' in matches_df.columns:
                        available_cols.extend(['window_start_date', 'window_end_date'])
                    elif 'start_date' in matches_df.columns:
                        available_cols.extend(['start_date', 'end_date'])
                    
                    display_df = matches_df[available_cols].head(10).copy()
                    display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x:.1%}")
                    display(HTML(display_df.to_html(index=False)))
                    
                else:
                    # No matches meet threshold - show top candidates anyway
                    print(f"\n⚠️  No patterns found meeting {min_confidence:.0%} confidence threshold")
                    print(f"📊 However, here are the top 10 candidates from {len(scan_list)} stocks scanned:")
                    
                    top_candidates = all_results.head(10)
                    print(f"🎯 Best candidate: {top_candidates.iloc[0]['ticker']} with {top_candidates.iloc[0]['confidence_score']:.1%} confidence")
                    
                    # Show top candidates table - use available columns
                    available_cols = ['ticker', 'confidence_score']
                    if 'window_start_date' in top_candidates.columns:
                        available_cols.extend(['window_start_date', 'window_end_date'])
                    elif 'start_date' in top_candidates.columns:
                        available_cols.extend(['start_date', 'end_date'])
                    
                    display_df = top_candidates[available_cols].copy()
                    display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x:.1%}")
                    display_df['status'] = 'Below Threshold'
                    
                    display(HTML(display_df.to_html(index=False)))
                    
                    print(f"\n💡 Suggestions:")
                    print(f"   • Lower confidence threshold to {top_candidates.iloc[0]['confidence_score']:.0%} to include top candidate")
                    print(f"   • Refine your positive/negative examples")
                    print(f"   • Try different date ranges for your pattern")
                    
            else:
                print(f"\n❌ No patterns found in {len(scan_list)} stocks scanned.")
                print("💡 Try adjusting your pattern examples or expanding the scan range.")
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
        find_similar_patterns_enhanced(
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


# %% [raw]
# ## 🎮 Enhanced Interactive Interface
#
# **UPDATED VERSION**: Clean logging, progress tracking, and improved UX
#

# %%
# ENHANCED USER INTERFACE with Smart Logging
def create_enhanced_interface():
    """Create improved interactive interface with better UX"""
    
    # Input widgets with better defaults and validation
    positive_ticker_input = widgets.Text(
        value='0700.HK',
        description='Positive Stock:',
        placeholder='e.g., 0700.HK',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    start_date_input = widgets.Text(
        value='2024-01-15',
        description='Pattern Start:',
        placeholder='YYYY-MM-DD',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    end_date_input = widgets.Text(
        value='2024-02-05',
        description='Pattern End:',
        placeholder='YYYY-MM-DD',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    negative_tickers_input = widgets.Textarea(
        value='0005.HK, 0941.HK, 0388.HK',
        description='Negative Examples:',
        placeholder='Comma-separated tickers (e.g., 0005.HK, 0001.HK)',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px', height='60px')
    )
    
    # Advanced configuration options
    confidence_slider = widgets.FloatSlider(
        value=0.7,
        min=0.5,
        max=0.95,
        step=0.05,
        description='Min Confidence:',
        style={'description_width': 'initial'},
        readout_format='.0%'
    )
    
    max_stocks_input = widgets.IntText(
        value=100,
        description='Max Stocks to Scan:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    # Action button
    run_button = widgets.Button(
        description="🔍 Find Similar Patterns",
        button_style='primary',
        layout=widgets.Layout(width='250px', height='40px'),
        tooltip='Start pattern scanning with current settings'
    )
    
    # Output area
    output_area = widgets.Output()
    
    # Status indicator
    status_html = widgets.HTML(
        value="<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>"
              "<b>📊 Status:</b> Ready to scan. Configure your pattern above and click 'Find Similar Patterns'.</div>"
    )
    
    def on_button_click(b):
        """Enhanced button click handler with validation"""
        with output_area:
            clear_output(True)
            
            # Update status
            status_html.value = ("<div style='padding: 10px; background-color: #fff3cd; border-radius: 5px;'>"
                               "<b>🔄 Status:</b> Scanning in progress... Please wait.</div>")
            
            try:
                # Simple validation
                if not positive_ticker_input.value.strip():
                    raise ValueError("Please enter a positive ticker symbol.")
                if not start_date_input.value.strip() or not end_date_input.value.strip():
                    raise ValueError("Please enter both start and end dates.")
                if not negative_tickers_input.value.strip():
                    raise ValueError("Please enter at least one negative example.")
                
                # For now, use the original function with enhanced display
                print("🔍 **ENHANCED PATTERN SCANNING**")
                print("=" * 50)
                print(f"✅ Suppressing verbose logs for cleaner output")
                print(f"✅ Progress tracking enabled")
                print(f"✅ Confidence threshold: {confidence_slider.value:.0%}")
                print(f"✅ Max stocks limit: {max_stocks_input.value}")
                print()
                
                # Call the enhanced function with clean logging
                find_similar_patterns_enhanced(
                    positive_ticker=positive_ticker_input.value.strip(),
                    start_date_str=start_date_input.value.strip(),
                    end_date_str=end_date_input.value.strip(),
                    negative_tickers_str=negative_tickers_input.value.strip(),
                    min_confidence=confidence_slider.value,
                    max_stocks_to_scan=max_stocks_input.value
                )
                
                # Update success status
                status_html.value = ("<div style='padding: 10px; background-color: #d1edff; border-radius: 5px;'>"
                                   "<b>✅ Status:</b> Pattern scanning completed successfully!</div>")
                
            except Exception as e:
                print(f"❌ **Input Error:** {str(e)}")
                status_html.value = ("<div style='padding: 10px; background-color: #f8d7da; border-radius: 5px;'>"
                                   f"<b>❌ Status:</b> Error - {str(e)}</div>")
    
    # Connect button to handler
    run_button.on_click(on_button_click)
    
    # Assemble interface
    interface = widgets.VBox([
        widgets.HTML("<h3>🎯 Enhanced Pattern Definition</h3>"),
        widgets.HTML("<p><b>Define one positive example of the pattern you want to find:</b></p>"),
        
        widgets.HBox([positive_ticker_input, start_date_input, end_date_input]),
        
        widgets.HTML("<br><p><b>Provide negative examples (stocks that DON'T show this pattern):</b></p>"),
        negative_tickers_input,
        
        widgets.HTML("<br><h3>⚙️ Enhanced Configuration</h3>"),
        widgets.HBox([confidence_slider, max_stocks_input]),
        
        widgets.HTML("<br>"),
        run_button,
        status_html,
        
        widgets.HTML("<br><h3>📊 Results</h3>"),
        output_area
    ])
    
    return interface

# Display the enhanced interface
print("🎮 **ENHANCED INTERFACE**: Clean logging, better validation, improved UX")
enhanced_interface = create_enhanced_interface()
display(enhanced_interface)


# %% [raw]
# ## 💡 Enhanced Features Summary
#
# ✅ **Smart Logging Control**: No more overwhelming output during large scans  
# ✅ **Progress Tracking**: Visual progress bars and performance metrics  
# ✅ **Better Interface**: Improved widgets with validation and guidance  
# ✅ **Configuration Options**: Confidence thresholds and scan limits  
# ✅ **Status Updates**: Real-time feedback during processing  
#
# **Note**: The enhanced interface above uses cleaner logging and better UX compared to the original version. The underlying pattern detection remains the same with improved presentation.
#

# %%

# %%

# %%
