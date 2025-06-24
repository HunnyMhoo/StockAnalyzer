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
# # 🧪 Quick Output Test - Fix Slow Notebook Issue
#
# This notebook tests the immediate output fix for Jupyter notebook cells that appear to "hang" without showing output.
#

# %%
# Test immediate output
import sys
import os
sys.path.append(os.path.join('..', 'src'))

from notebook_utils import quick_test_notebook_output, notebook_friendly_fetch_demo

print("🧪 Testing immediate output...")
sys.stdout.flush()

# This should show output immediately
quick_test_notebook_output()


# %%
# Test the improved data fetching with immediate output
from data_fetcher import fetch_hk_stocks
from datetime import datetime, timedelta

print("🚀 Testing improved data fetcher...")
sys.stdout.flush()

# Test with just one stock and a short date range
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

print(f"📅 Date range: {start_date} to {end_date}")
print(f"🎯 Testing with 1 stock: 0700.HK (Tencent)")
print(f"💡 You should see output immediately as it processes!")
sys.stdout.flush()

# This should show immediate output for each step
test_data = fetch_hk_stocks(
    tickers=['0700.HK'],
    start_date=start_date,
    end_date=end_date
)

print(f"\n✅ Test completed! Got {len(test_data)} stocks")
for ticker, data in test_data.items():
    print(f"📊 {ticker}: {len(data)} records")


# %%
# Test bulk fetching with immediate output
from bulk_data_fetcher import fetch_hk_stocks_bulk

print("🚀 Testing bulk fetcher with immediate output...")
print("💡 Each batch should show progress immediately!")
sys.stdout.flush()

# Test with 3 stocks in small batches
test_stocks = ['0700.HK', '0005.HK', '0941.HK']

bulk_data = fetch_hk_stocks_bulk(
    tickers=test_stocks,
    start_date=start_date,
    end_date=end_date,
    batch_size=2,           # Small batches to see immediate progress
    delay_between_batches=0.5,  # Short delay for demo
    force_refresh=False
)

print(f"\n🎉 Bulk test completed! Got {len(bulk_data)} stocks")
print("💡 Notice how you saw each step immediately as it happened!")

