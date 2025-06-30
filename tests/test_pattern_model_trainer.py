"""
Unit tests for pattern model trainer module.

Tests the PatternModelTrainer class and related functionality to ensure
proper model training pipeline and error handling.
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open

# Handle imports for both direct execution and package usage
try:
    from stock_analyzer.analysis import (
        PatternModelTrainer,
        TrainingConfig,
        TrainingResults,
        ModelTrainingError,
        load_trained_model,
        quick_train_model
    )
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from stock_analyzer.analysis import (
        PatternModelTrainer,
        TrainingConfig,
        TrainingResults,
        ModelTrainingError,
        load_trained_model,
        quick_train_model
    )


class TestTrainingConfig:
    """Test class for TrainingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.model_type == "xgboost"
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.use_cross_validation is True
        assert config.cv_folds == 5
        assert config.apply_smote is True
        assert config.scale_features is False
        assert config.save_model is True
        assert config.model_name is None
        assert config.overwrite_existing is False
        assert isinstance(config.hyperparameters, dict)
        assert len(config.hyperparameters) == 0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_params = {'n_estimators': 200, 'max_depth': 6}
        config = TrainingConfig(
            model_type="randomforest",
            test_size=0.3,
            random_state=123,
            hyperparameters=custom_params,
            model_name="custom_model"
        )
        
        assert config.model_type == "randomforest"
        assert config.test_size == 0.3
        assert config.random_state == 123
        assert config.hyperparameters == custom_params
        assert config.model_name == "custom_model"


class TestPatternModelTrainer:
    """Test class for PatternModelTrainer."""
    
    @pytest.fixture
    def sample_features_data(self):
        """Create sample features dataset for testing."""
        np.random.seed(42)
        n_samples = 50
        
        # Create sample features
        features = {
            'prior_trend_return': np.random.normal(0, 5, n_samples),
            'above_sma_50_ratio': np.random.uniform(0, 10, n_samples),
            'trend_angle': np.random.normal(0, 2, n_samples),
            'drawdown_pct': np.random.uniform(-20, 0, n_samples),
            'recovery_return_pct': np.random.uniform(0, 15, n_samples),
            'down_day_ratio': np.random.uniform(0, 100, n_samples),
            'support_level': np.random.uniform(50, 200, n_samples),
            'support_break_depth_pct': np.random.uniform(0, 10, n_samples),
            'false_break_flag': np.random.randint(0, 2, n_samples),
            'recovery_days': np.random.randint(0, 20, n_samples),
            'recovery_volume_ratio': np.random.uniform(0.5, 2.0, n_samples),
            'sma_5': np.random.uniform(50, 200, n_samples),
            'sma_10': np.random.uniform(50, 200, n_samples),
            'sma_20': np.random.uniform(50, 200, n_samples),
            'rsi_14': np.random.uniform(20, 80, n_samples),
            'macd_diff': np.random.normal(0, 1, n_samples),
            'volatility': np.random.uniform(0, 0.1, n_samples),
            'volume_avg_ratio': np.random.uniform(0.5, 2.0, n_samples)
        }
        
        # Add metadata columns
        features['ticker'] = [f'TEST{i:02d}.HK' for i in range(n_samples)]
        features['start_date'] = ['2023-01-01'] * n_samples
        features['end_date'] = ['2023-01-31'] * n_samples
        features['notes'] = ['Test pattern'] * n_samples
        
        # Create labels with class imbalance (70% negative, 30% positive)
        labels = np.random.choice(['negative', 'positive'], n_samples, p=[0.7, 0.3])
        features['label_type'] = labels
        
        return pd.DataFrame(features)
    
    @pytest.fixture
    def trainer_with_temp_dir(self):
        """Create trainer instance with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            features_file = os.path.join(temp_dir, "test_features.csv")
            models_dir = os.path.join(temp_dir, "models")
            
            config = TrainingConfig(
                model_type="randomforest",  # Use RF to avoid XGBoost dependency issues
                save_model=False  # Don't save models in tests
            )
            
            trainer = PatternModelTrainer(
                features_file=features_file,
                models_dir=models_dir,
                config=config
            )
            
            yield trainer, features_file, temp_dir
    
    def test_trainer_initialization(self):
        """Test PatternModelTrainer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(model_type="randomforest")
            trainer = PatternModelTrainer(
                features_file="test.csv",
                models_dir=temp_dir,
                config=config
            )
            
            assert trainer.features_file == "test.csv"
            assert trainer.models_dir == temp_dir
            assert trainer.config.model_type == "randomforest"
            assert os.path.exists(temp_dir)
    
    def test_config_validation_invalid_model_type(self):
        """Test configuration validation with invalid model type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(model_type="invalid_model")
            
            with pytest.raises(ModelTrainingError, match="Unsupported model type"):
                PatternModelTrainer(
                    features_file="test.csv",
                    models_dir=temp_dir,
                    config=config
                )
    
    def test_config_validation_invalid_test_size(self):
        """Test configuration validation with invalid test size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(test_size=1.5)
            
            with pytest.raises(ModelTrainingError, match="test_size must be between 0 and 1"):
                PatternModelTrainer(
                    features_file="test.csv",
                    models_dir=temp_dir,
                    config=config
                )
    
    def test_config_validation_invalid_cv_folds(self):
        """Test configuration validation with invalid CV folds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(cv_folds=1)
            
            with pytest.raises(ModelTrainingError, match="cv_folds must be >= 2"):
                PatternModelTrainer(
                    features_file="test.csv",
                    models_dir=temp_dir,
                    config=config
                )
    
    def test_load_and_validate_data_file_not_found(self, trainer_with_temp_dir):
        """Test data loading with missing file."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        with pytest.raises(ModelTrainingError, match="Features file not found"):
            trainer.load_and_validate_data()
    
    def test_load_and_validate_data_empty_file(self, trainer_with_temp_dir, sample_features_data):
        """Test data loading with empty file."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Create empty CSV file
        empty_df = pd.DataFrame()
        empty_df.to_csv(features_file, index=False)
        
        with pytest.raises(ModelTrainingError, match="Features file is empty"):
            trainer.load_and_validate_data()
    
    def test_load_and_validate_data_insufficient_samples(self, trainer_with_temp_dir):
        """Test data loading with insufficient samples."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Create minimal dataset
        small_data = pd.DataFrame({
            'feature1': [1, 2],
            'label_type': ['positive', 'negative']
        })
        small_data.to_csv(features_file, index=False)
        
        with pytest.raises(ModelTrainingError, match="Insufficient samples"):
            trainer.load_and_validate_data()
    
    def test_load_and_validate_data_missing_label_column(self, trainer_with_temp_dir):
        """Test data loading with missing label column."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Create dataset without label_type column
        data = pd.DataFrame({
            'feature1': np.random.randn(40),
            'feature2': np.random.randn(40)
        })
        data.to_csv(features_file, index=False)
        
        with pytest.raises(ModelTrainingError, match="Missing required columns"):
            trainer.load_and_validate_data()
    
    def test_load_and_validate_data_no_positive_samples(self, trainer_with_temp_dir):
        """Test data loading with no positive samples."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Create dataset with only negative samples
        data = pd.DataFrame({
            'feature1': np.random.randn(40),
            'label_type': ['negative'] * 40
        })
        data.to_csv(features_file, index=False)
        
        with pytest.raises(ModelTrainingError, match="No positive samples found"):
            trainer.load_and_validate_data()
    
    def test_load_and_validate_data_success(self, trainer_with_temp_dir, sample_features_data):
        """Test successful data loading and validation."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Save sample data
        sample_features_data.to_csv(features_file, index=False)
        
        # Load and validate
        loaded_data = trainer.load_and_validate_data()
        
        assert len(loaded_data) == len(sample_features_data)
        assert 'label_type' in loaded_data.columns
        assert 'positive' in loaded_data['label_type'].values
    
    def test_prepare_features_and_labels(self, trainer_with_temp_dir, sample_features_data):
        """Test feature and label preparation."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        features, labels = trainer.prepare_features_and_labels(sample_features_data)
        
        # Check that non-feature columns are excluded
        excluded_columns = ['ticker', 'start_date', 'end_date', 'label_type', 'notes']
        for col in excluded_columns:
            assert col not in features.columns
        
        # Check label encoding
        assert labels.dtype == int
        assert set(labels.unique()).issubset({0, 1})
        
        # Check that positive labels are encoded as 1
        positive_mask = sample_features_data['label_type'] == 'positive'
        assert all(labels[positive_mask] == 1)
        assert all(labels[~positive_mask] == 0)
    
    def test_split_data(self, trainer_with_temp_dir, sample_features_data):
        """Test data splitting functionality."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        features, labels = trainer.prepare_features_and_labels(sample_features_data)
        X_train, X_test, y_train, y_test = trainer.split_data(features, labels)
        
        # Check split proportions
        total_samples = len(features)
        expected_test_size = int(total_samples * trainer.config.test_size)
        
        assert len(X_test) == expected_test_size
        assert len(X_train) == total_samples - expected_test_size
        assert len(y_test) == expected_test_size
        assert len(y_train) == total_samples - expected_test_size
        
        # Check that all samples are accounted for
        assert len(X_train) + len(X_test) == total_samples
        assert len(y_train) + len(y_test) == total_samples
    
    def test_create_model_randomforest(self, trainer_with_temp_dir):
        """Test Random Forest model creation."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        model = trainer.create_model()
        
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == trainer.config.random_state
    
    def test_create_model_with_hyperparameters(self, trainer_with_temp_dir):
        """Test model creation with custom hyperparameters."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Set custom hyperparameters
        trainer.config.hyperparameters = {
            'n_estimators': 50,
            'max_depth': 5
        }
        
        model = trainer.create_model()
        
        assert model.n_estimators == 50
        assert model.max_depth == 5
    
    def test_apply_preprocessing_no_smote_no_scaling(self, trainer_with_temp_dir, sample_features_data):
        """Test preprocessing without SMOTE or scaling."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Disable SMOTE and scaling
        trainer.config.apply_smote = False
        trainer.config.scale_features = False
        
        features, labels = trainer.prepare_features_and_labels(sample_features_data)
        X_train, X_test, y_train, y_test = trainer.split_data(features, labels)
        
        X_train_proc, X_test_proc, y_train_proc, scaler = trainer.apply_preprocessing(
            X_train, X_test, y_train
        )
        
        # Should return original data
        pd.testing.assert_frame_equal(X_train_proc, X_train)
        pd.testing.assert_frame_equal(X_test_proc, X_test)
        pd.testing.assert_series_equal(y_train_proc, y_train)
        assert scaler is None
    
    def test_apply_preprocessing_with_scaling(self, trainer_with_temp_dir, sample_features_data):
        """Test preprocessing with feature scaling."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Enable scaling, disable SMOTE
        trainer.config.apply_smote = False
        trainer.config.scale_features = True
        
        features, labels = trainer.prepare_features_and_labels(sample_features_data)
        X_train, X_test, y_train, y_test = trainer.split_data(features, labels)
        
        X_train_proc, X_test_proc, y_train_proc, scaler = trainer.apply_preprocessing(
            X_train, X_test, y_train
        )
        
        # Check that scaler was created and applied
        assert scaler is not None
        from sklearn.preprocessing import StandardScaler
        assert isinstance(scaler, StandardScaler)
        
        # Check that data was scaled (means should be close to 0)
        assert abs(X_train_proc.mean().mean()) < 0.1
    
    @patch('src.pattern_model_trainer.SMOTE')
    def test_apply_preprocessing_with_smote(self, mock_smote, trainer_with_temp_dir, sample_features_data):
        """Test preprocessing with SMOTE."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Enable SMOTE, disable scaling
        trainer.config.apply_smote = True
        trainer.config.scale_features = False
        
        # Mock SMOTE
        mock_smote_instance = MagicMock()
        mock_smote.return_value = mock_smote_instance
        
        features, labels = trainer.prepare_features_and_labels(sample_features_data)
        X_train, X_test, y_train, y_test = trainer.split_data(features, labels)
        
        # Mock SMOTE to return balanced data
        balanced_X = np.vstack([X_train.values, X_train.values[:5]])  # Add 5 samples
        balanced_y = np.hstack([y_train.values, np.ones(5)])  # Add 5 positive samples
        mock_smote_instance.fit_resample.return_value = (balanced_X, balanced_y)
        
        X_train_proc, X_test_proc, y_train_proc, scaler = trainer.apply_preprocessing(
            X_train, X_test, y_train
        )
        
        # Check that SMOTE was called
        mock_smote_instance.fit_resample.assert_called_once()
        
        # Check that balanced data is returned
        assert len(X_train_proc) == len(balanced_X)
        assert len(y_train_proc) == len(balanced_y)
    
    def test_extract_feature_importance(self, trainer_with_temp_dir, sample_features_data):
        """Test feature importance extraction."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Create a mock model with feature importances
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.1, 0.3, 0.2, 0.4])
        
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4']
        
        importance_df = trainer.extract_feature_importance(mock_model, feature_names)
        
        assert importance_df is not None
        assert len(importance_df) == 4
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        
        # Check that features are sorted by importance (descending)
        assert importance_df.iloc[0]['feature'] == 'feature4'  # Highest importance
        assert importance_df.iloc[0]['importance'] == 0.4
    
    def test_extract_feature_importance_no_support(self, trainer_with_temp_dir):
        """Test feature importance extraction with unsupported model."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Create a mock model without feature importances
        mock_model = MagicMock()
        del mock_model.feature_importances_  # Remove the attribute
        
        feature_names = ['feature1', 'feature2']
        
        importance_df = trainer.extract_feature_importance(mock_model, feature_names)
        
        assert importance_df is None
    
    @patch('src.pattern_model_trainer.joblib.dump')
    def test_save_model(self, mock_joblib_dump, trainer_with_temp_dir):
        """Test model saving functionality."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Enable model saving
        trainer.config.save_model = True
        trainer.config.model_name = "test_model"
        
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        feature_names = ['feature1', 'feature2']
        metadata = {'test': 'data'}
        
        model_path = trainer.save_model(mock_model, mock_scaler, feature_names, metadata)
        
        # Check that joblib.dump was called
        mock_joblib_dump.assert_called_once()
        
        # Check model package structure
        call_args = mock_joblib_dump.call_args[0]
        model_package = call_args[0]
        
        assert 'model' in model_package
        assert 'scaler' in model_package
        assert 'feature_names' in model_package
        assert 'config' in model_package
        assert 'metadata' in model_package
        
        assert model_package['model'] == mock_model
        assert model_package['scaler'] == mock_scaler
        assert model_package['feature_names'] == feature_names
        assert model_package['metadata'] == metadata
    
    def test_save_model_disabled(self, trainer_with_temp_dir):
        """Test that model saving can be disabled."""
        trainer, features_file, temp_dir = trainer_with_temp_dir
        
        # Disable model saving
        trainer.config.save_model = False
        
        mock_model = MagicMock()
        model_path = trainer.save_model(mock_model, None, [], {})
        
        assert model_path == ""


class TestModelLoading:
    """Test class for model loading functionality."""
    
    @patch('src.pattern_model_trainer.joblib.load')
    def test_load_trained_model_success(self, mock_joblib_load):
        """Test successful model loading."""
        # Mock model package
        mock_package = {
            'model': MagicMock(),
            'feature_names': ['feature1', 'feature2'],
            'config': TrainingConfig(),
            'metadata': {'training_date': '2023-01-01'}
        }
        mock_joblib_load.return_value = mock_package
        
        with patch('os.path.exists', return_value=True):
            loaded_package = load_trained_model('test_model.pkl')
        
        assert loaded_package == mock_package
        mock_joblib_load.assert_called_once_with('test_model.pkl')
    
    def test_load_trained_model_file_not_found(self):
        """Test model loading with missing file."""
        with pytest.raises(ModelTrainingError, match="Model file not found"):
            load_trained_model('nonexistent_model.pkl')
    
    @patch('src.pattern_model_trainer.joblib.load')
    def test_load_trained_model_invalid_structure(self, mock_joblib_load):
        """Test model loading with invalid model package structure."""
        # Mock incomplete model package
        mock_package = {
            'model': MagicMock(),
            # Missing required keys
        }
        mock_joblib_load.return_value = mock_package
        
        with patch('os.path.exists', return_value=True):
            with pytest.raises(ModelTrainingError, match="Invalid model file"):
                load_trained_model('test_model.pkl')


class TestQuickTrainModel:
    """Test class for quick_train_model convenience function."""
    
    @patch('src.pattern_model_trainer.PatternModelTrainer')
    def test_quick_train_model(self, mock_trainer_class):
        """Test quick training function."""
        # Mock trainer instance and training results
        mock_trainer = MagicMock()
        mock_results = MagicMock()
        mock_trainer.train.return_value = mock_results
        mock_trainer_class.return_value = mock_trainer
        
        result = quick_train_model(
            features_file="test.csv",
            model_type="randomforest",
            test_size=0.3
        )
        
        # Check that trainer was created with correct parameters
        mock_trainer_class.assert_called_once()
        call_args = mock_trainer_class.call_args
        
        assert call_args[1]['features_file'] == "test.csv"
        
        # Check that train was called
        mock_trainer.train.assert_called_once()
        assert result == mock_results


class TestTrainingResults:
    """Test class for TrainingResults dataclass."""
    
    def test_training_results_creation(self):
        """Test TrainingResults dataclass creation."""
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        config = TrainingConfig()
        
        results = TrainingResults(
            model=mock_model,
            scaler=mock_scaler,
            config=config,
            train_score={'accuracy': 0.95},
            test_score={'accuracy': 0.90},
            cv_scores={'cv_accuracy_mean': 0.92},
            feature_importance=None,
            model_path="/path/to/model.pkl",
            training_time=10.5,
            metadata={'samples': 100}
        )
        
        assert results.model == mock_model
        assert results.scaler == mock_scaler
        assert results.config == config
        assert results.train_score['accuracy'] == 0.95
        assert results.test_score['accuracy'] == 0.90
        assert results.cv_scores['cv_accuracy_mean'] == 0.92
        assert results.feature_importance is None
        assert results.model_path == "/path/to/model.pkl"
        assert results.training_time == 10.5
        assert results.metadata['samples'] == 100


# Integration tests
class TestIntegration:
    """Integration tests for the complete training pipeline."""
    
    @pytest.fixture
    def integration_setup(self):
        """Setup for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample dataset
            np.random.seed(42)
            n_samples = 60
            
            # Ensure we have enough samples for both classes
            n_positive = 20
            n_negative = 40
            
            data = {
                'ticker': [f'TEST{i:02d}.HK' for i in range(n_samples)],
                'start_date': ['2023-01-01'] * n_samples,
                'end_date': ['2023-01-31'] * n_samples,
                'notes': ['Test pattern'] * n_samples,
                'feature1': np.random.normal(0, 1, n_samples),
                'feature2': np.random.normal(0, 1, n_samples),
                'feature3': np.random.uniform(0, 10, n_samples),
                'feature4': np.random.exponential(1, n_samples),
                'label_type': ['positive'] * n_positive + ['negative'] * n_negative
            }
            
            df = pd.DataFrame(data)
            features_file = os.path.join(temp_dir, 'features.csv')
            df.to_csv(features_file, index=False)
            
            yield temp_dir, features_file, df
    
    def test_end_to_end_training_randomforest(self, integration_setup):
        """Test complete training pipeline with Random Forest."""
        temp_dir, features_file, df = integration_setup
        
        config = TrainingConfig(
            model_type='randomforest',
            test_size=0.2,
            use_cross_validation=False,  # Disable CV for faster testing
            apply_smote=False,  # Disable SMOTE for simpler testing
            save_model=False  # Don't save models in tests
        )
        
        trainer = PatternModelTrainer(
            features_file=features_file,
            models_dir=temp_dir,
            config=config
        )
        
        # This should complete without errors
        results = trainer.train()
        
        # Check results structure
        assert isinstance(results, TrainingResults)
        assert results.model is not None
        assert 'accuracy' in results.train_score
        assert 'accuracy' in results.test_score
        assert results.training_time > 0
        
        # Check that scores are reasonable
        assert 0 <= results.train_score['accuracy'] <= 1
        assert 0 <= results.test_score['accuracy'] <= 1
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient training data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dataset with insufficient samples
            small_data = pd.DataFrame({
                'feature1': [1, 2, 3],
                'label_type': ['positive', 'negative', 'positive']
            })
            
            features_file = os.path.join(temp_dir, 'small_features.csv')
            small_data.to_csv(features_file, index=False)
            
            config = TrainingConfig(model_type='randomforest')
            trainer = PatternModelTrainer(
                features_file=features_file,
                models_dir=temp_dir,
                config=config
            )
            
            with pytest.raises(ModelTrainingError, match="Insufficient samples"):
                trainer.train()


if __name__ == "__main__":
    pytest.main([__file__])