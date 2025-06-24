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
# # Pattern Labeling Demo
#
# This notebook demonstrates how to manually label stock chart patterns for training data.
#
# ## Features
# - Define labeled patterns with ticker, date range, and pattern type
# - Validate input data and store in JSON format
# - Load and manage existing labels
# - Optional visualization of labeled patterns
#
# ## Prerequisites
# - Story 1.1 completed (OHLCV data fetching)
# - Required libraries: pandas, json, datetime
# - Optional: mplfinance for chart visualization
#

# %% [raw]
# ## Setup and Imports
#

# %%
# Standard imports
import sys
import os
import json
from datetime import datetime
from typing import List, Dict

# Add src to path for imports
sys.path.append('../src')

# Import pattern labeling functionality
from pattern_labeler import (
    PatternLabeler, 
    PatternLabel, 
    LabelValidator,
    save_labeled_patterns, 
    load_labeled_patterns,
    ValidationError,
    PatternLabelError
)

# Try to import visualization (optional)
try:
    from pattern_visualizer import display_labeled_pattern, compare_patterns
    VISUALIZATION_AVAILABLE = True
    print("✓ Visualization available")
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"⚠️  Visualization not available: {e}")

print("✓ Pattern labeling modules imported successfully")


# %% [raw]
# ## Initialize Pattern Labeler
#

# %%
# Initialize the pattern labeler
labeler = PatternLabeler()
print(f"Pattern labeler initialized. Labels file: {labeler.labels_file}")


# %% [raw]
# ## Example: Define Pattern Labels Using Dictionary Format
#
# This matches the format specified in the user story.
#

# %%
# Example labeled patterns from user story
labeled_patterns = [
    {
        "ticker": "0700.HK",
        "start_date": "2023-02-10",
        "end_date": "2023-03-03",
        "label_type": "positive",
        "notes": "Classic false breakdown before breakout"
    },
    {
        "ticker": "0005.HK",
        "start_date": "2022-10-15",
        "end_date": "2022-11-01",
        "label_type": "positive",
        "notes": "High volume recovery zone"
    }
]

print("Example patterns defined:")
for pattern in labeled_patterns:
    print(f"  - {pattern['ticker']}: {pattern['start_date']} to {pattern['end_date']} ({pattern['label_type']})")


# %% [raw]
# ## Save Patterns Using the Convenience Function
#

# %%
# Ensure labeled_patterns is defined (in case previous cell wasn't run)
if 'labeled_patterns' not in locals():
    labeled_patterns = [
        {
            "ticker": "0700.HK",
            "start_date": "2023-02-10",
            "end_date": "2023-03-03",
            "label_type": "positive",
            "notes": "Classic false breakdown before breakout"
        },
        {
            "ticker": "0005.HK",
            "start_date": "2022-10-15",
            "end_date": "2022-11-01",
            "label_type": "positive",
            "notes": "High volume recovery zone"
        }
    ]
    print("ℹ️  Re-defined labeled_patterns (previous cell may not have been executed)")

# Save the labeled patterns using the convenience function
try:
    save_labeled_patterns(labeled_patterns)
    print("✓ Patterns saved successfully using convenience function")
except (ValidationError, PatternLabelError) as e:
    print(f"❌ Error saving patterns: {e}")


# %% [raw]
# ## Use the PatternLabeler Instance to Add More Patterns
#

# %%
# Ensure labeler is defined (in case previous cell wasn't run)
if 'labeler' not in locals():
    labeler = PatternLabeler()
    print("ℹ️  Re-initialized labeler (previous cell may not have been executed)")

# Now use the labeler instance to add additional patterns
try:
    # Add a new positive pattern using the labeler
    new_label = labeler.add_label(
        ticker="0001.HK",
        start_date="2023-01-15",
        end_date="2023-02-05",
        label_type="positive",
        notes="Strong volume breakout pattern"
    )
    
    # Add a negative example using the labeler
    negative_label = labeler.add_label(
        ticker="0388.HK",
        start_date="2022-12-01",
        end_date="2022-12-20",
        label_type="negative",
        notes="Failed breakout, lack of volume confirmation"
    )
    
    print("✓ Additional labels added successfully using labeler instance")
    
except (ValidationError, PatternLabelError) as e:
    print(f"❌ Error adding labels: {e}")


# %% [raw]
# ## Load and Display All Patterns Using the Labeler
#

# %%
# Ensure labeler is available
if 'labeler' not in locals():
    labeler = PatternLabeler()
    print("ℹ️  Re-initialized labeler (previous cell may not have been executed)")

# Load all patterns using the labeler instance
try:
    all_patterns = labeler.load_labels()
    print(f"✓ Loaded {len(all_patterns)} patterns using labeler:")
    
    for i, pattern in enumerate(all_patterns, 1):
        print(f"\n{i}. {pattern.ticker}")
        print(f"   Period: {pattern.start_date} to {pattern.end_date}")
        print(f"   Type: {pattern.label_type}")
        print(f"   Notes: {pattern.notes}")
        print(f"   Created: {pattern.created_at}")
        
except PatternLabelError as e:
    print(f"❌ Error loading patterns: {e}")


# %% [raw]
# ## Use Labeler for Summary Statistics and Filtering
#

# %%
# Ensure labeler is available
if 'labeler' not in locals():
    labeler = PatternLabeler()
    print("ℹ️  Re-initialized labeler (previous cell may not have been executed)")

# Use labeler to get summary statistics
summary = labeler.get_labels_summary()
print("📊 Pattern Label Summary using labeler:")
print(f"   Total labels: {summary['total_labels']}")
print(f"   Unique tickers: {summary['unique_tickers']}")
print(f"   Positive labels: {summary['positive_labels']}")
print(f"   Negative labels: {summary['negative_labels']}")
print(f"   Neutral labels: {summary['neutral_labels']}")

print("\n🔍 Filter patterns by ticker using labeler:")
# Get all labels for a specific ticker using labeler
hk700_labels = labeler.get_labels_by_ticker("0700.HK")
print(f"\nLabels for 0700.HK:")
for label in hk700_labels:
    print(f"  Period: {label.start_date} to {label.end_date}")
    print(f"  Type: {label.label_type}")
    print(f"  Notes: {label.notes}")


# %% [raw]
# ## Demonstrate Validation Using Labeler
#

# %%
# Ensure labeler is available
if 'labeler' not in locals():
    labeler = PatternLabeler()
    print("ℹ️  Re-initialized labeler (previous cell may not have been executed)")

# Demonstrate validation errors using labeler
print("Testing validation errors with labeler:\n")

# Test invalid ticker format
try:
    labeler.add_label(
        ticker="INVALID",
        start_date="2023-01-01",
        end_date="2023-01-15"
    )
except ValidationError as e:
    print(f"✓ Caught invalid ticker: {e}\n")

# Test invalid date range
try:
    labeler.add_label(
        ticker="0700.HK",
        start_date="2023-01-15",
        end_date="2023-01-01"  # End before start
    )
except ValidationError as e:
    print(f"✓ Caught invalid date range: {e}\n")

# Test duplicate detection using labeler
try:
    # This should fail because we already have a label for 0700.HK with these dates
    labeler.add_label(
        ticker="0700.HK",
        start_date="2023-02-10",
        end_date="2023-03-03",
        overwrite=False  # Don't allow overwrite
    )
except PatternLabelError as e:
    print(f"✓ Caught duplicate label: {e}\n")


# %%

# %%
