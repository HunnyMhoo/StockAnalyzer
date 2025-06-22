### User Story 1.4 – Train a Pattern Detection Model

**User Story**  
As a trader, I want to train a supervised machine learning model using my labeled feature data, so that the system can learn to recognize my preferred pattern and detect it in new stock charts.

---

**Context**  
After labeling pattern windows and extracting features, I need to train a binary classifier that can distinguish between pattern and non-pattern segments. Since this is a personal tool, the training pipeline should be easy to run in a Jupyter notebook and provide clear outputs such as accuracy, confidence scores, and false-positive examples for review.

---

**Priority**  
Must Have

---

**Functional Requirements**

1. **Input Requirements**
   - Labeled feature dataset from `./features/labeled_features.csv`
   - Required columns:
     - All numerical features extracted in Story 1.3
     - `label_type` (must include at least one class = 'positive')

2. **Model Type**
   - Binary classification model (initially `XGBoostClassifier` or `RandomForestClassifier`)
   - Trained to predict whether a new segment is likely to match the labeled pattern

3. **Pipeline Steps**
   - Split data into training and validation sets (e.g. 80/20 stratified split)
   - Normalize features if needed (optional for tree-based models)
   - Train classifier on training set
   - Evaluate using:
     - Accuracy, precision, recall, F1-score
     - Confusion matrix
     - Optional: ROC AUC and feature importance plot
   - Save model as `.pkl` to `./models/` (e.g., `model_xgb_v1.pkl`)

4. **Model Output**
   - Trained classifier object
   - Probability scores (0 to 1) for each test sample
   - Ranked list of validation samples by confidence
   - Feature importance (ranked list or bar chart)

---

**Non-Functional Requirements**

- Training script must run in under 10 seconds for <200 samples
- Model file must be portable (pickle or joblib)
- Notebook output should include:
  - Key metrics
  - Misclassified examples
  - Chart of feature importance (if available)
- Retraining should not overwrite previous models unless `overwrite=True`

---

**Acceptance Criteria**

1. Given a labeled dataset with at least 30 samples, the classifier trains successfully and achieves measurable performance on a holdout set.
2. Training produces a saved model file at `./models/model_xgb_v1.pkl` (or similar).
3. Console/log output includes confusion matrix and precision/recall.
4. Probability scores are available for each validation example.
5. Model handles missing or malformed labels with clear error messages.
6. Feature importance list is generated and highlights at least 3 significant features.

---

**Dependencies**

- Requires:
  - **Story 1.3 – Feature Extraction**
- Libraries:
  - `pandas`, `sklearn`, `xgboost` or `lightgbm`, `joblib`, `matplotlib` (for optional plots)
