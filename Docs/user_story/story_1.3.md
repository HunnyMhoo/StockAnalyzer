### User Story 1.3 – Extract Features from Labeled Data

**User Story**  
As a trader, I want to extract numerical features from my labeled pattern windows, including those that capture trend strength, correction behavior, and false support breaks, so that I can train a model to detect my preferred bullish setup.

---

**Context**  
I identify patterns manually by spotting stocks that are in a bullish trend, temporarily pull back, and then falsely break a prior support level before recovering. To teach a model to detect these setups, I need each labeled segment to be converted into a fixed-length feature vector that reflects this logic.

---

**Priority**  
Must Have

---

**Functional Requirements**

1. **Inputs:**
   - Labeled pattern list from `labeled_patterns.json` or Python variable
   - OHLCV data from `./data/{ticker}.csv`
   - Configurable window size (default: 30 days per pattern)

2. **Output:**
   - One row per labeled pattern
   - Format: Pandas DataFrame with shape `(num_samples, num_features)`
   - Save to: `./features/labeled_features.csv`

3. **Core Feature Set**
   Each row will contain the following types of engineered features:

   **A. Trend Context Features:**
   - `prior_trend_return`: % return from day -30 to day -1 before window
   - `above_sma_50_ratio`: % of days in prior 30-day period where close > 50-day SMA
   - `trend_angle`: linear regression slope of closing price in prior 30 days

   **B. Correction Phase Features:**
   - `drawdown_pct`: max % drop from recent high during current window
   - `recovery_return_pct`: return from drawdown low to final day
   - `down_day_ratio`: % of red candles in first half of window

   **C. False Support Break Features:**
   - `support_level`: minimum low from last 10 days before window
   - `support_break_depth_pct`: % below support the lowest low dropped
   - `false_break_flag`: 1 if price broke below support but closed above it within 3 bars
   - `recovery_days`: number of days from support break to recovery close
   - `recovery_volume_ratio`: volume on recovery candle / 20-day average volume

   **D. General Price/Volume Indicators (Final Day):**
   - `sma_5`, `sma_10`, `sma_20`, `rsi_14`, `macd_diff`
   - `volatility`: standard deviation of returns
   - `volume_avg_ratio`: volume on last day / 20-day average

4. **Metadata Columns:**
   - `ticker`, `start_date`, `end_date`, `label_type`, `notes` (if any)

---

**Non-Functional Requirements**

- Each feature must return float or boolean, with no missing values
- Incomplete segments (e.g. fewer than 30 candles) are skipped with a warning
- Calculations must be reproducible and encapsulated in functions
- Feature extractor must allow new fields to be added easily
- Chart preview (optional) can help verify support break detection

---

**Acceptance Criteria**

1. At least 10 engineered features are extracted for every valid labeled pattern.
2. The final output CSV contains one row per pattern with consistent schema.
3. Missing data (e.g., incomplete historical context or malformed ticker) results in a skipped row with a warning.
4. `false_break_flag` correctly identifies a break below support and recovery within 3 days (confirmed in test cases).
5. `support_level` is accurately computed as the lowest low in the 10 days before the window.
6. `prior_trend_return` reflects strong positive return (>5%) in at least 50% of labeled samples.

---

**Dependencies**

- Completion of:
  - **Story 1.1 – Data Fetching**
  - **Story 1.2 – Pattern Labeling**
- Libraries:
  - `pandas`, `numpy`, `scipy` (for local minima), `ta-lib` or `custom indicators`, `sklearn` (for trend regression)
