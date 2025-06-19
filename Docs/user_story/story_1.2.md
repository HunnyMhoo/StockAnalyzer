### User Story 1.2 – Manually Label Pattern Setup Periods

**User Story**  
As a trader, I want to manually label historical stock chart segments that match my preferred pattern, so that I can provide examples for the model to learn from.

---

**Context**  
The user (me) has a personal strategy for recognizing bullish continuation patterns. Rather than using fixed rules, I will define examples by specifying a stock ticker and a date range where the setup occurred. These will serve as supervised training data. Labeling will be done in Jupyter via Python or JSON and saved locally.

---

**Priority**  
Must Have

---

**Functional Requirements**

1. **Input Format:**  
   Each label entry must include:
   - `ticker` (string, e.g., `'0700.HK'`)
   - `start_date` (string, `'YYYY-MM-DD'`)
   - `end_date` (string, `'YYYY-MM-DD'`)
   - `label_type` (optional, default `'positive'`)
   - `notes` (optional, string)

2. **Label Entry Examples (Python format):**

   ```python
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
   ```

3. **Persistence:**
   - Labels are stored in a file: `./labels/labeled_patterns.json`
   - Notebook includes helper functions:
     - `save_labeled_patterns(label_list, path)`
     - `load_labeled_patterns(path)`

4. **Optional Visualization Tool:**
   - Display a candlestick chart with volume for any labeled entry
   - Highlight the labeled window using color shading or annotations

---

**Non-Functional Requirements**

- Duplicate entries should either warn or overwrite based on config
- Validate `start_date` and `end_date` format; ensure logical order
- Handle empty list inputs or malformed entries gracefully

---

**Acceptance Criteria**

1. At least 5 labeled patterns can be defined and stored.
2. `labeled_patterns.json` is created with the correct schema.
3. JSON file includes: `ticker`, `start_date`, `end_date`, and optional `label_type`, `notes`.
4. Loading the JSON into memory does not raise exceptions.
5. Chart preview tool correctly displays the time window and overlays the pattern range.
6. Providing an invalid date range raises a clear and actionable error message.

---

**Dependencies**

- Requires completion of: **Story 1.1 – Fetch and Cache OHLCV Data**
- Required libraries: `pandas`, `json`, `datetime`, and optionally `mplfinance` for charts
