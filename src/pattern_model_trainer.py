"""
Pattern Model Trainer for Stock Trading Pattern Detection

This module provides the PatternModelTrainer class that implements a complete
machine learning pipeline for training binary classifiers to detect trading patterns.
"""

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")


# Configuration constants
MODELS_DIR = "models"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
MIN_SAMPLES_REQUIRED = 30
SUPPORTED_MODEL_TYPES = ["xgboost", "randomforest"]


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    pass


@dataclass
class TrainingConfig:
    """
    Configuration class for model training parameters.
    
    Attributes:
        model_type: Type of model to train ('xgboost' or 'randomforest')
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        use_cross_validation: Whether to perform cross-validation
        cv_folds: Number of cross-validation folds
        apply_smote: Whether to apply SMOTE for class balancing
        scale_features: Whether to scale features (mainly for non-tree models)
        hyperparameters: Model-specific hyperparameters
        save_model: Whether to save the trained model
        model_name: Custom name for the saved model
        overwrite_existing: Whether to overwrite existing model files
    """
    model_type: str = "xgboost"
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    use_cross_validation: bool = True
    cv_folds: int = 5
    apply_smote: bool = True
    scale_features: bool = False
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    save_model: bool = True
    model_name: Optional[str] = None
    overwrite_existing: bool = False


@dataclass
class TrainingResults:
    """
    Results container for model training outcomes.
    
    Attributes:
        model: Trained model object
        scaler: Fitted scaler (if used)
        config: Training configuration used
        train_score: Training set performance
        test_score: Test set performance
        cv_scores: Cross-validation scores (if performed)
        feature_importance: Feature importance rankings
        model_path: Path to saved model file
        training_time: Time taken to train the model
        metadata: Additional training metadata
    """
    model: Any
    scaler: Optional[StandardScaler]
    config: TrainingConfig
    train_score: Dict[str, float]
    test_score: Dict[str, float]
    cv_scores: Optional[Dict[str, float]]
    feature_importance: Optional[pd.DataFrame]
    model_path: Optional[str]
    training_time: float
    metadata: Dict[str, Any]


class PatternModelTrainer:
    """
    Main class for training pattern detection models.
    
    This class provides a complete machine learning pipeline including:
    - Data loading and validation
    - Preprocessing and feature scaling
    - Model training with multiple algorithms
    - Cross-validation and performance evaluation
    - Model persistence and metadata tracking
    """
    
    def __init__(self, 
                 features_file: str = "features/labeled_features.csv",
                 models_dir: str = MODELS_DIR,
                 config: Optional[TrainingConfig] = None):
        """
        Initialize PatternModelTrainer.
        
        Args:
            features_file: Path to labeled features CSV file
            models_dir: Directory for saving trained models
            config: Training configuration (uses defaults if None)
        """
        self.features_file = features_file
        self.models_dir = models_dir
        self.config = config or TrainingConfig()
        self._ensure_models_directory()
        self._validate_config()
    
    def _ensure_models_directory(self) -> None:
        """Create models directory if it doesn't exist."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"✓ Created models directory: {self.models_dir}/")
    
    def _validate_config(self) -> None:
        """Validate training configuration parameters."""
        if self.config.model_type not in SUPPORTED_MODEL_TYPES:
            raise ModelTrainingError(
                f"Unsupported model type: {self.config.model_type}. "
                f"Supported types: {SUPPORTED_MODEL_TYPES}"
            )
        
        if self.config.model_type == "xgboost" and not XGBOOST_AVAILABLE:
            raise ModelTrainingError(
                "XGBoost not available. Install with: pip install xgboost"
            )
        
        if not 0 < self.config.test_size < 1:
            raise ModelTrainingError(
                f"test_size must be between 0 and 1, got: {self.config.test_size}"
            )
        
        if self.config.cv_folds < 2:
            raise ModelTrainingError(
                f"cv_folds must be >= 2, got: {self.config.cv_folds}"
            )
    
    def _train_model_instance(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Helper to fit a model instance, created to be called by other methods."""
        try:
            if self.config.model_type == "xgboost" and XGBOOST_AVAILABLE:
                print("🚀 Training xgboost model...")
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            raise ModelTrainingError(f"Model fitting failed: {e}")

    def train_model(self, features_df: pd.DataFrame, labels: pd.Series) -> Any:
        """
        Train a model directly on a provided DataFrame and labels.
        This is a convenience wrapper for interactive use.
        """
        # This method is for direct, in-memory training. It doesn't use the full pipeline.
        if not all(features_df.dtypes.apply(pd.api.types.is_numeric_dtype)):
            raise ModelTrainingError("All columns in features_df must be numeric.")
        
        model = self.create_model()
        return self._train_model_instance(model, features_df, labels)
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Load and validate the labeled features dataset.
        
        Returns:
            pd.DataFrame: Validated features dataset
            
        Raises:
            ModelTrainingError: If data loading or validation fails
        """
        try:
            # Load data
            if not os.path.exists(self.features_file):
                raise ModelTrainingError(f"Features file not found: {self.features_file}")
            
            data = pd.read_csv(self.features_file)
            
            # Basic validation
            if data.empty:
                raise ModelTrainingError("Features file is empty")
            
            if len(data) < MIN_SAMPLES_REQUIRED:
                raise ModelTrainingError(
                    f"Insufficient samples: {len(data)} < {MIN_SAMPLES_REQUIRED} required"
                )
            
            # Check for required columns
            required_columns = ['label_type']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ModelTrainingError(f"Missing required columns: {missing_columns}")
            
            # Validate label types
            if 'positive' not in data['label_type'].values:
                raise ModelTrainingError("No positive samples found in label_type column")
            
            print(f"✓ Loaded {len(data)} samples from {self.features_file}")
            print(f"✓ Label distribution:\n{data['label_type'].value_counts()}")
            
            return data
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Failed to load data: {e}")
    
    def prepare_features_and_labels(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target labels from the dataset.
        
        Args:
            data: Raw labeled features dataset
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        # Exclude non-feature columns
        exclude_columns = [
            'ticker', 'start_date', 'end_date', 'label_type', 'notes'
        ]
        
        # Get feature columns
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        features = data[feature_columns].copy()
        
        # Create binary labels (1 for positive, 0 for others)
        labels = (data['label_type'] == 'positive').astype(int)
        
        # Handle missing values
        if features.isnull().any().any():
            print("⚠ Found missing values in features. Filling with median values.")
            features = features.fillna(features.median())
        
        print(f"✓ Prepared {len(feature_columns)} features for {len(features)} samples")
        print(f"✓ Class distribution: {labels.value_counts().to_dict()}")
        
        return features, labels
    
    def split_data(self, 
                   features: pd.DataFrame, 
                   labels: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets with stratification.
        
        Args:
            features: Feature matrix
            labels: Target labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=labels
            )
            
            print(f"✓ Split data: {len(X_train)} train, {len(X_test)} test samples")
            print(f"✓ Train class distribution: {y_train.value_counts().to_dict()}")
            print(f"✓ Test class distribution: {y_test.value_counts().to_dict()}")
            
            return X_train, X_test, y_train, y_test
            
        except ValueError as e:
            # Handle cases where stratification fails due to class imbalance
            if "least populated class" in str(e):
                print("⚠ Stratification failed due to class imbalance. Using random split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state
                )
                return X_train, X_test, y_train, y_test
            raise ModelTrainingError(f"Failed to split data: {e}")
    
    def apply_preprocessing(self, 
                           X_train: pd.DataFrame, 
                           X_test: pd.DataFrame, 
                           y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Optional[StandardScaler]]:
        """
        Apply preprocessing steps including scaling and SMOTE.
        
        Args:
            X_train: Training features
            X_test: Test features  
            y_train: Training labels
            
        Returns:
            Tuple of (X_train_processed, X_test_processed, y_train_processed, scaler)
        """
        scaler = None
        
        # Feature scaling (if requested)
        if self.config.scale_features:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            print("✓ Applied feature scaling")
        else:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
        
        # Apply SMOTE for class balancing (if requested)
        if self.config.apply_smote:
            try:
                smote = SMOTE(random_state=self.config.random_state)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                
                # Convert back to DataFrame/Series with proper index
                X_train_balanced = pd.DataFrame(
                    X_train_balanced,
                    columns=X_train_scaled.columns
                )
                y_train_balanced = pd.Series(y_train_balanced)
                
                print(f"✓ Applied SMOTE: {len(X_train_scaled)} → {len(X_train_balanced)} samples")
                print(f"✓ Balanced class distribution: {y_train_balanced.value_counts().to_dict()}")
                
                return X_train_balanced, X_test_scaled, y_train_balanced, scaler
                
            except Exception as e:
                print(f"⚠ SMOTE failed: {e}. Proceeding without class balancing.")
        
        return X_train_scaled, X_test_scaled, y_train, scaler
    
    def create_model(self) -> Any:
        """
        Create and configure the specified model type.
        
        Returns:
            Configured model instance
        """
        if self.config.model_type == "xgboost":
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.config.random_state,
                'eval_metric': 'logloss'
            }
            params = {**default_params, **self.config.hyperparameters}
            return xgb.XGBClassifier(**params)
        
        elif self.config.model_type == "randomforest":
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': self.config.random_state
            }
            params = {**default_params, **self.config.hyperparameters}
            return RandomForestClassifier(**params)
        
        else:
            raise ModelTrainingError(f"Unsupported model type: {self.config.model_type}")
    
    def evaluate_model(self, 
                       model: Any, 
                       X_train: pd.DataFrame, 
                       y_train: pd.Series,
                       X_test: pd.DataFrame, 
                       y_test: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate model performance on training and test sets.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tuple of (train_scores, test_scores)
        """
        def calculate_scores(y_true, y_pred, y_pred_proba):
            """Calculate comprehensive performance scores."""
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
            }
        
        # Training set evaluation
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_scores = calculate_scores(y_train, y_train_pred, y_train_proba)
        
        # Test set evaluation
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_scores = calculate_scores(y_test, y_test_pred, y_test_proba)
        
        print("📊 Model Performance:")
        print(f"   Training Accuracy: {train_scores['accuracy']:.3f}")
        print(f"   Test Accuracy: {test_scores['accuracy']:.3f}")
        print(f"   Test Precision: {test_scores['precision']:.3f}")
        print(f"   Test Recall: {test_scores['recall']:.3f}")
        print(f"   Test F1-Score: {test_scores['f1_score']:.3f}")
        print(f"   Test ROC-AUC: {test_scores['roc_auc']:.3f}")
        
        return train_scores, test_scores
    
    def perform_cross_validation(self, 
                                 model: Any, 
                                 X: pd.DataFrame, 
                                 y: pd.Series) -> Dict[str, float]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            
        Returns:
            Dictionary of CV scores
        """
        if not self.config.use_cross_validation:
            return {}
        
        print(f"🔄 Performing {self.config.cv_folds}-fold cross-validation...")
        
        try:
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds, 
                shuffle=True, 
                random_state=self.config.random_state
            )
            
            # Calculate multiple metrics
            cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
            cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall') 
            cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
            
            cv_scores = {
                'cv_accuracy_mean': cv_accuracy.mean(),
                'cv_accuracy_std': cv_accuracy.std(),
                'cv_precision_mean': cv_precision.mean(),
                'cv_precision_std': cv_precision.std(),
                'cv_recall_mean': cv_recall.mean(),
                'cv_recall_std': cv_recall.std(),
                'cv_f1_mean': cv_f1.mean(),
                'cv_f1_std': cv_f1.std()
            }
            
            print(f"✓ CV Accuracy: {cv_scores['cv_accuracy_mean']:.3f} ± {cv_scores['cv_accuracy_std']:.3f}")
            print(f"✓ CV F1-Score: {cv_scores['cv_f1_mean']:.3f} ± {cv_scores['cv_f1_std']:.3f}")
            
            return cv_scores
            
        except Exception as e:
            print(f"⚠ Cross-validation failed: {e}")
            return {}
    
    def extract_feature_importance(self, 
                                   model: Any, 
                                   feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Extract and rank feature importance from the trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance rankings
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("🎯 Top 5 Most Important Features:")
                for _, row in importance_df.head().iterrows():
                    print(f"   {row['feature']}: {row['importance']:.4f}")
                
                return importance_df
                
        except Exception as e:
            print(f"⚠ Failed to extract feature importance: {e}")
        
        return None
    
    def save_model(self, 
                   model: Any, 
                   scaler: Optional[StandardScaler], 
                   feature_names: List[str],
                   metadata: Dict[str, Any]) -> str:
        """
        Save trained model and associated artifacts.
        
        Args:
            model: Trained model
            scaler: Fitted scaler (if used)
            feature_names: List of feature names
            metadata: Training metadata
            
        Returns:
            Path to saved model file
        """
        if not self.config.save_model:
            return ""
        
        # Generate model filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.config.model_name:
            model_filename = f"{self.config.model_name}.pkl"
        else:
            model_filename = f"model_{self.config.model_type}_{timestamp}.pkl"
        
        model_path = os.path.join(self.models_dir, model_filename)
        
        # Check if file exists and handle overwrite policy
        if os.path.exists(model_path) and not self.config.overwrite_existing:
            model_filename = f"model_{self.config.model_type}_{timestamp}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
        
        # Prepare model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'config': self.config,
            'metadata': metadata
        }
        
        try:
            joblib.dump(model_package, model_path)
            print(f"💾 Model saved to: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"⚠ Failed to save model: {e}")
            return ""
    
    def train(self) -> TrainingResults:
        """
        Execute the complete training pipeline.
        """
        start_time = datetime.now()
        
        try:
            print("🎯 Starting Pattern Model Training Pipeline")
            print("=" * 50)
            
            # Load and validate data
            data = self.load_and_validate_data()
            
            # Prepare features and labels
            features, labels = self.prepare_features_and_labels(data)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(features, labels)
            
            # Apply preprocessing
            X_train_processed, X_test_processed, y_train_processed, scaler = self.apply_preprocessing(
                X_train, X_test, y_train
            )
            
            # Train model
            model = self.create_model()
            model = self._train_model_instance(model, X_train_processed, y_train_processed)
            
            # Evaluate model
            training_time = (datetime.now() - start_time).total_seconds()
            train_scores, test_scores = self.evaluate_model(
                model, X_train_processed, y_train_processed, X_test_processed, y_test
            )
            
            # Cross-validation
            cv_scores = self.perform_cross_validation(model, X_train_processed, y_train_processed) if self.config.use_cross_validation else None
            
            # Feature importance
            feature_importance = self.extract_feature_importance(model, list(features.columns))
            
            # Prepare metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'training_time_seconds': training_time,
                'data_file': self.features_file,
                'n_samples': len(data),
                'n_features': len(features.columns),
                'class_distribution': labels.value_counts().to_dict()
            }
            
            # Save model
            model_path = self.save_model(model, scaler, list(features.columns), metadata) if self.config.save_model else None
            
            print("=" * 50)
            print("✅ Training Pipeline Completed Successfully!")
            print(f"⏱ Total Time: {training_time:.2f} seconds")
            
            return TrainingResults(
                model=model,
                scaler=scaler,
                config=self.config,
                train_score=train_scores,
                test_score=test_scores,
                cv_scores=cv_scores,
                feature_importance=feature_importance,
                model_path=model_path,
                training_time=training_time,
                metadata=metadata
            )
            
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Training pipeline failed: {e}")


def load_trained_model(model_path: str) -> Dict[str, Any]:
    """
    Load a previously trained model and its artifacts.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dictionary containing model and associated artifacts
        
    Raises:
        ModelTrainingError: If loading fails
    """
    try:
        if not os.path.exists(model_path):
            raise ModelTrainingError(f"Model file not found: {model_path}")
        
        model_package = joblib.load(model_path)
        
        # Validate model package structure
        required_keys = ['model', 'feature_names', 'config', 'metadata']
        missing_keys = [key for key in required_keys if key not in model_package]
        if missing_keys:
            raise ModelTrainingError(f"Invalid model file. Missing keys: {missing_keys}")
        
        print(f"✓ Loaded model from: {model_path}")
        print(f"✓ Model type: {model_package['config'].model_type}")
        print(f"✓ Training date: {model_package['metadata'].get('training_date', 'Unknown')}")
        
        return model_package
        
    except Exception as e:
        if isinstance(e, ModelTrainingError):
            raise
        raise ModelTrainingError(f"Failed to load model: {e}")


def quick_train_model(features_file: str = "features/labeled_features.csv",
                     model_type: str = "xgboost",
                     **kwargs) -> TrainingResults:
    """
    Convenience function for quick model training with default settings.
    
    Args:
        features_file: Path to labeled features file
        model_type: Type of model to train
        **kwargs: Additional configuration parameters
        
    Returns:
        TrainingResults object
    """
    config = TrainingConfig(model_type=model_type, **kwargs)
    trainer = PatternModelTrainer(features_file=features_file, config=config)
    return trainer.train()