# Stock Pattern Recognition Engine – Personal MVP Spec

## 1. Product Overview

A self-hosted, personal tool to automate my manual chart pattern recognition process. The system learns from my own labeled examples (bullish continuation setups) and scans Hong Kong stocks for similar patterns using daily price and volume data.

## 2. Goals

- Automate daily stock screening based on my custom logic
- Identify stocks showing early signs of my preferred setup
- Reduce time spent on chart-by-chart manual review
- Improve over time via feedback and retraining

## 3. Scope (Phase 1)

| Capability                              | Included |
|-----------------------------------------|----------|
| Fetch HK stock OHLCV data               | Yes      |
| Manually label historical chart patterns| Yes      |
| Extract technical features              | Yes      |
| Train a pattern detection model         | Yes      |
| Run detection across new data           | Yes      |
| Output match list (e.g. CSV or console) | Yes      |
| Plot chart with match overlay (optional)| Optional |
| Track pattern outcome (optional)        | Optional |

## 4. Core Workflow

1. **Label Patterns**
   - Input: ticker + start_date + end_date
   - Store as JSON or CSV for training

2. **Train Model**
   - Extract features (price trend, volume behavior, etc.)
   - Train ML model (e.g. XGBoost) on labeled windows

3. **Run Detection**
   - Scan new daily data across HK stock universe
   - Output windows that match trained pattern logic

4. **Review Matches**
   - Optionally visualize charts
   - Track performance and relabel as needed

## 5. Technical Stack

| Layer     | Technology                      |
|-----------|----------------------------------|
| Data      | yfinance / EOD Historical Data  |
| Storage   | CSV / JSON (local filesystem)   |
| Features  | pandas, numpy, TA-Lib           |
| Modeling  | scikit-learn, XGBoost           |
| Charts    | mplfinance                      |
| UI        | CLI / Jupyter Notebook          |
| Automation| Manual CLI or cron (optional)   |

## 6. MVP Feature List

### Must Have
- Manual input for labeled examples
- Feature extraction logic
- Model training pipeline
- Daily scanning against unlabeled stocks
- CLI or CSV output of matches

### Optional
- Chart viewer for detected signals
- Feedback tagging (worked / failed)
- Signal history and backtesting module

## 7. Daily Usage Flow

- Run script to update HK stock data
- Apply model to generate new candidate matches
- Review top-ranked matches
- Manually inspect and confirm
- Optionally tag outcome and retrain over time

## 8. Success Criteria

- Accurately flags 70% of patterns I would find manually
- Reduces chart review time by over 50%
- Detects early-stage patterns not visible yet
- Operates entirely locally with minimal manual work
- Scales well as I label more data over time
