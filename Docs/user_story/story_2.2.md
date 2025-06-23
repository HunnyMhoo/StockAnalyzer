### User Story 2.2 – Tag Signal Outcome (Success or Failure)

**User Story**  
As a trader, I want to manually tag whether each predicted match resulted in a successful pattern outcome, so that I can track accuracy, analyze false positives, and improve future model training through feedback.

---

**Context**  
After detecting and reviewing matches, I often know within a few days or weeks whether a prediction turned out to be correct (e.g. breakout happened or not). This story introduces a feedback mechanism to **label the real-world result** of each prediction, so that the system can evolve and improve using actual outcomes.

---

**Priority**  
Should Have

---

**Functional Requirements**

1. **Inputs**
   - Match file from `./signals/matches_YYYYMMDD.csv` with:
     - `ticker`, `window_start_date`, `confidence_score`
   - Outcome tag from user:
     - `outcome`: one of `'success'`, `'failure'`, `'uncertain'`
     - Optional: `feedback_notes` (e.g. "false breakout after 3 days")

2. **Tagging Logic**
   - Load the match file into a DataFrame
   - Allow tagging individual matches manually (by index, or by `ticker+date`)
   - Store outcome and notes in two new columns:
     - `outcome`, `feedback_notes`
   - Save the updated DataFrame to:
     - `./signals/matches_YYYYMMDD_labeled.csv`

3. **Feedback Review Option**
   - Provide notebook cell to:
     - Filter by outcome (e.g. show all `failure`)
     - Group and summarize outcomes by confidence band (e.g. 0.7–0.8, 0.8–0.9)

---

**Non-Functional Requirements**

- Feedback tagging must not overwrite original files unless explicitly confirmed
- Support incremental feedback sessions (edit same file over time)
- Interface must be simple — pandas cell or short-form CLI input
- Feedback can be later merged into training data via new feature `source = feedback`

**Notebook Cell Validation Requirements**
- Each feedback cell must:
  - Confirm target match exists before applying label
  - Validate `outcome` value and flag any invalid entries
  - Print confirmation of tag applied and where it’s saved
- All tagging operations should be reversible or overwrite-safe

---

**Acceptance Criteria**

1. A user can load any past match file and apply manual outcome labels.
2. Feedback is saved in a new or versioned file in `./signals/`.
3. All entries include correct values for `outcome` and (optional) `feedback_notes`.
4. A helper function allows review of tagging statistics (e.g. success rate by score range).
5. Tagging works even if only partial matches are labeled in one session.
6. Attempting to tag a nonexistent match results in a helpful error.

---

**Dependencies**

- Requires:
  - Story 1.5 – Match file available
  - Story 2.1 – Optional chart validation to support tagging
- Libraries:
  - `pandas`, `os`, `datetime`, optionally `ipywidgets` or `input()` for CLI
