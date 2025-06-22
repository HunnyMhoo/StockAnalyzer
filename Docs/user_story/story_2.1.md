### User Story 2.1 – Visualize Detected Matches

**User Story**  
As a trader, I want to visualize the candlestick chart for each detected pattern match, with key overlays like support levels and detection windows, so that I can manually verify the quality of the prediction and decide whether to track the stock.

---

**Context**  
After scanning for matches using the trained model, I need to visually inspect the raw chart data to confirm that the pattern looks correct. This serves both as a sanity check and an insight loop for improving labels or tuning the model. The system should render each match with clear visual cues on the detected window, key levels, and volume trends.

---

**Priority**  
Should Have

---

**Functional Requirements**

1. **Inputs**
   - A `matches.csv` file from `./signals/` with at least:
     - `ticker`, `window_start_date`, `window_end_date`, `confidence_score`
   - OHLCV data from `./data/{ticker}.csv`
   - Optional: `support_level` (if precomputed), label notes (if available)

2. **Chart Generation Logic**
   - Load OHLCV data and isolate a range around the matched window
     - e.g. 10 days before + 30-day detection window + 5 days after
   - Plot a candlestick chart using `mplfinance` or `plotly`
   - Overlay:
     - Detection window (highlighted region)
     - Support level (horizontal line, optional)
     - Volume bar chart below

3. **Output**
   - Inline chart in Jupyter notebook
   - Optional: Save chart to `./charts/{ticker}_{window_start}.png`
   - Ability to visualize one match or loop through all matches with preview

---

**Non-Functional Requirements**

- Chart generation must complete in under 1 second per match
- Visualization logic must not crash if data is missing; should skip with warning
- Chart annotations must be clear, readable, and labeled
- Charting functions should be modular (e.g. `plot_match(match_row)`)

**Notebook Cell Validation Requirements**
- Each chart function must:
  - Assert that data exists for full chart window range
  - Print metadata for the match (e.g., score, window dates)
  - Include visual sanity checks (e.g. `plt.show()` confirmation, or `fig.savefig()` path)
  - Handle missing volume or out-of-bounds date windows gracefully

---

**Acceptance Criteria**

1. For each row in the match file, the chart displays:
   - Correct time range and candles
   - Detected window highlighted
   - Optional support line and volume bars
2. A helper function allows rendering:
   - Single match by index or `ticker + date`
   - All matches sequentially (with optional filter by score)
3. Chart is saved locally if `save=True` is passed
4. Any failures (e.g. data not found, plotting error) result in warnings, not crashes
5. Console output prints metadata per chart for review

---

**Dependencies**

- Requires:
  - Story 1.5 – Detection match file must be available
  - Story 1.1 – OHLCV data must be cached
- Libraries:
  - `pandas`, `mplfinance` or `plotly`, `matplotlib`, `os`, `datetime`
