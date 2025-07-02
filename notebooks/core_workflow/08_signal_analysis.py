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
# # Signal Outcome Tagging Notebook
# ## User Story 2.2 - Tag Signal Outcome (Success or Failure)
#
# This notebook provides an interactive interface for manually tagging pattern match outcomes
# to track prediction accuracy and improve future model training through feedback loops.
#
# ### Key Features:
# - Load and review pattern match files from `./signals/`
# - Tag individual matches with outcomes: success, failure, uncertain
# - Add detailed feedback notes for each prediction
# - Batch tagging capabilities for efficient processing
# - Feedback analysis and performance review by confidence bands
# - Safe file operations with automatic backups
#
# ### Prerequisites:
# - Pattern match files generated from User Story 1.5 (Pattern Scanning)
# - Matches stored in `./signals/matches_YYYYMMDD.csv` format
#

# %%
# Import required libraries
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append('../src')

# Import signal outcome tagging functionality
from signal_outcome_tagger import (
    SignalOutcomeTagger,
    SignalOutcomeError,
    load_latest_matches,
    quick_tag_outcome,
    review_latest_feedback,
    VALID_OUTCOMES
)

print("📦 Signal Outcome Tagging System Loaded")
print(f"   Valid outcomes: {', '.join(VALID_OUTCOMES)}")
print("   Ready for pattern match feedback collection!")


# %% [raw]
# ## 🔍 Step 1: Discover Available Match Files
#
# First, let's check what pattern match files are available for tagging.
#

# %%
# Initialize the tagger and discover available files
signals_dir = '../signals'

try:
    tagger = SignalOutcomeTagger(signals_dir=signals_dir)
    match_files = tagger.find_available_match_files()
    
    print("📊 Available Match Files for Tagging:")
    print("=" * 50)
    
    if not match_files:
        print("❌ No match files found!")
        print("   Please run pattern scanning (User Story 1.5) first to generate matches.")
    else:
        for i, file_path in enumerate(match_files, 1):
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            # Quick peek at file content
            try:
                temp_df = pd.read_csv(file_path)
                match_count = len(temp_df)
                confidence_range = f"{temp_df['confidence_score'].min():.3f} - {temp_df['confidence_score'].max():.3f}"
                
                print(f"   {i}. {filename}")
                print(f"      Matches: {match_count}, Confidence: {confidence_range}")
                print(f"      Size: {file_size:,} bytes")
                
            except Exception as e:
                print(f"   {i}. {filename} (Error reading: {e})")
        
        print(f"\n✅ Found {len(match_files)} match files ready for tagging")
        
except SignalOutcomeError as e:
    print(f"❌ Error initializing tagger: {e}")
    print("   Please ensure the signals directory exists and contains match files.")


# %% [raw]
# ## 📂 Step 2: Load and Review Match File
#
# Load the most recent match file and review its contents.
#

# %%
# Load the latest match file
try:
    file_path, matches_df = load_latest_matches(signals_dir)
    
    print(f"📄 Loaded Match File: {os.path.basename(file_path)}")
    print("=" * 60)
    
    # Display basic statistics
    summary = tagger.get_match_summary(matches_df)
    
    print(f"📊 Match File Summary:")
    print(f"   Total matches: {summary['total_matches']}")
    print(f"   Already tagged: {summary['tagged_count']}")
    print(f"   Untagged: {summary['untagged_count']}")
    print(f"   Confidence range: {summary['confidence_range'][0]:.3f} - {summary['confidence_range'][1]:.3f}")
    print(f"   Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"   Tickers: {', '.join(summary['tickers'])}")
    
    # Display the matches for review
    print(f"\n🎯 Pattern Matches:")
    display_columns = ['ticker', 'window_start_date', 'window_end_date', 'confidence_score', 'rank']
    
    # Add outcome columns if they exist
    if 'outcome' in matches_df.columns:
        display_columns.extend(['outcome', 'feedback_notes'])
    
    display_df = matches_df[display_columns].copy()
    
    # Format confidence scores for better readability
    display_df['confidence_score'] = display_df['confidence_score'].round(3)
    
    print(display_df.to_string(index=False, max_colwidth=20))
    
    print(f"\n✅ Match file loaded successfully!")
    print(f"   Ready for outcome tagging.")
    
except SignalOutcomeError as e:
    print(f"❌ Error loading matches: {e}")
    matches_df = None
    file_path = None


# %% [raw]
# ## 🏷️ Step 3: Individual Match Tagging
#
# Tag individual matches with outcomes and feedback notes.
#
# ### Usage Instructions:
# 1. Identify the match you want to tag from the table above
# 2. Use the ticker and window_start_date to uniquely identify it
# 3. Choose outcome: 'success', 'failure', or 'uncertain'
# 4. Add optional feedback notes explaining your reasoning
#

# %%
# Example: Tag a single match outcome
# Modify these parameters to tag your specific matches

if matches_df is not None:
    # Configuration for tagging - MODIFY THESE VALUES
    ticker_to_tag = '0005.HK'  # Change to your target ticker
    window_start_to_tag = '2024-08-19'  # Change to your target date
    outcome_to_apply = 'success'  # Choose: 'success', 'failure', 'uncertain'
    feedback_notes = 'Strong breakout after support test - volume confirmed the move'
    
    try:
        # Apply the tag
        updated_matches_df = tagger.tag_outcome(
            matches_df,
            ticker=ticker_to_tag,
            window_start_date=window_start_to_tag,
            outcome=outcome_to_apply,
            feedback_notes=feedback_notes,
            overwrite=False  # Set to True to overwrite existing tags
        )
        
        print(f"✅ Successfully tagged {ticker_to_tag} ({window_start_to_tag}) as '{outcome_to_apply}'")
        
        # Update our working DataFrame
        matches_df = updated_matches_df
        
        # Show the updated row
        tagged_row = matches_df[
            (matches_df['ticker'] == ticker_to_tag) & 
            (matches_df['window_start_date'] == window_start_to_tag)
        ]
        
        if not tagged_row.empty:
            print("\n📋 Tagged Match Details:")
            row = tagged_row.iloc[0]
            print(f"   Ticker: {row['ticker']}")
            print(f"   Period: {row['window_start_date']} to {row['window_end_date']}")
            print(f"   Confidence: {row['confidence_score']:.3f}")
            print(f"   Outcome: {row['outcome']}")
            print(f"   Notes: {row['feedback_notes']}")
            print(f"   Tagged: {row['tagged_date']}")
        
    except SignalOutcomeError as e:
        print(f"❌ Error tagging outcome: {e}")
        print("   Check ticker and date values, or use overwrite=True for existing tags")
else:
    print("⚠️  No matches loaded. Please run the previous cell to load match data.")


# %% [raw]
# ## 💾 Step 4: Save Tagged Matches
#
# Save your tagged matches to a labeled file for future reference and analysis.
#

# %%
# Save the labeled matches
if matches_df is not None and file_path is not None:
    try:
        # Save labeled matches with automatic naming
        output_path = tagger.save_labeled_matches(matches_df, file_path)
        
        print(f"✅ Labeled matches saved successfully!")
        print(f"   Output file: {os.path.basename(output_path)}")
        
        # Verify saved content
        saved_df = pd.read_csv(output_path)
        tagged_count = (~saved_df['outcome'].isna()).sum()
        
        print(f"\n📊 Saved File Summary:")
        print(f"   Total matches: {len(saved_df)}")
        print(f"   Tagged matches: {tagged_count}")
        print(f"   Untagged matches: {len(saved_df) - tagged_count}")
        
        if tagged_count > 0:
            outcome_counts = saved_df['outcome'].value_counts()
            print(f"\n🎯 Outcome Distribution:")
            for outcome, count in outcome_counts.items():
                percentage = (count / tagged_count * 100)
                print(f"   {outcome.title()}: {count} ({percentage:.1f}%)")
        
    except SignalOutcomeError as e:
        print(f"❌ Error saving labeled matches: {e}")
else:
    print("⚠️  No matches to save. Please load and tag matches first.")


# %% [raw]
# ## 📈 Step 5: Feedback Analysis and Review
#
# Analyze your tagging results to understand model performance across different confidence bands.
#

# %%
# Analyze feedback results
if matches_df is not None:
    print("📊 Feedback Analysis")
    print("=" * 50)
    
    # Get detailed feedback analysis
    feedback_results = tagger.review_feedback(matches_df)
    
    # Additional insights
    if feedback_results['tagged_matches'] > 0:
        print("\n🔍 Additional Insights:")
        
        # Success rate by ticker (if multiple tickers)
        tagged_matches = matches_df[matches_df['outcome'].notna()]
        
        if len(tagged_matches['ticker'].unique()) > 1:
            print("\n📋 Performance by Ticker:")
            ticker_performance = tagged_matches.groupby('ticker')['outcome'].value_counts().unstack(fill_value=0)
            
            for ticker in ticker_performance.index:
                success = ticker_performance.loc[ticker].get('success', 0)
                failure = ticker_performance.loc[ticker].get('failure', 0)
                total_decisive = success + failure
                
                if total_decisive > 0:
                    success_rate = success / total_decisive
                    print(f"   {ticker}: {success_rate:.1%} success rate ({success}S/{failure}F)")
        
        # Recommendations based on results
        print("\n💡 Recommendations:")
        
        overall_success_rate = feedback_results['outcome_summary']['counts'].get('success', 0) / max(1, feedback_results['tagged_matches'])
        
        if overall_success_rate > 0.7:
            print("   ✅ Model performing well - consider lowering confidence threshold")
        elif overall_success_rate < 0.5:
            print("   ⚠️  Model needs improvement - consider raising confidence threshold")
        else:
            print("   📊 Model performance is moderate - continue collecting feedback")
        
        if feedback_results['tagged_matches'] < 20:
            print("   📈 Tag more matches for better statistical significance")
    
    else:
        print("\n💡 No tagged matches found for analysis.")
        print("   Tag some matches using the cells above to see feedback analysis.")
else:
    print("⚠️  No matches loaded for analysis.")

