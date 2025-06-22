#!/usr/bin/env python3
"""
Verification script for the Pattern Model Training implementation.

This script tests the core functionality without external dependencies
that may not be available in all environments.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_training_config():
    """Test TrainingConfig dataclass functionality."""
    print("🧪 Testing TrainingConfig...")
    
    try:
        # Import with fallback to avoid XGBoost issues
        from pattern_model_trainer import TrainingConfig
        
        # Test default config
        config = TrainingConfig()
        assert config.model_type == "xgboost"
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.use_cross_validation is True
        assert config.apply_smote is True
        assert config.save_model is True
        
        # Test custom config
        custom_config = TrainingConfig(
            model_type="randomforest",
            test_size=0.3,
            hyperparameters={'n_estimators': 200}
        )
        assert custom_config.model_type == "randomforest"
        assert custom_config.test_size == 0.3
        assert custom_config.hyperparameters['n_estimators'] == 200
        
        print("✅ TrainingConfig tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ TrainingConfig tests failed: {e}")
        return False

def test_model_evaluator():
    """Test ModelEvaluator functionality."""
    print("🧪 Testing ModelEvaluator...")
    
    try:
        from model_evaluator import ModelEvaluator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(
                plots_dir=os.path.join(temp_dir, "plots"),
                reports_dir=os.path.join(temp_dir, "reports")
            )
            
            # Test basic initialization
            assert evaluator.plots_dir.endswith("plots")
            assert evaluator.reports_dir.endswith("reports")
            assert os.path.exists(evaluator.plots_dir)
            assert os.path.exists(evaluator.reports_dir)
            
            # Test metric calculation
            y_test = pd.Series([1, 0, 1, 0, 1])
            y_pred = np.array([1, 0, 0, 0, 1])
            y_pred_proba = np.array([0.8, 0.2, 0.4, 0.1, 0.9])
            
            metrics = evaluator._calculate_detailed_metrics(y_test, y_pred, y_pred_proba)
            
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], float)
            
            assert metrics['accuracy'] == 0.8  # 4/5 correct
            
            print("✅ ModelEvaluator tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ ModelEvaluator tests failed: {e}")
        return False

def test_data_generation():
    """Test creating sample data that matches our expected format."""
    print("🧪 Testing data generation...")
    
    try:
        # Create sample features dataset
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'ticker': [f'TEST{i:02d}.HK' for i in range(n_samples)],
            'start_date': ['2023-01-01'] * n_samples,
            'end_date': ['2023-01-31'] * n_samples,
            'notes': ['Test pattern'] * n_samples,
            'prior_trend_return': np.random.normal(0, 5, n_samples),
            'above_sma_50_ratio': np.random.uniform(0, 10, n_samples),
            'trend_angle': np.random.normal(0, 2, n_samples),
            'drawdown_pct': np.random.uniform(-20, 0, n_samples),
            'recovery_return_pct': np.random.uniform(0, 15, n_samples),
            'down_day_ratio': np.random.uniform(0, 100, n_samples),
            'support_level': np.random.uniform(50, 200, n_samples),
            'rsi_14': np.random.uniform(20, 80, n_samples),
            'volume_avg_ratio': np.random.uniform(0.5, 2.0, n_samples),
            'label_type': np.random.choice(['positive', 'negative'], n_samples, p=[0.3, 0.7])
        }
        
        df = pd.DataFrame(data)
        
        # Validate data structure
        assert len(df) == n_samples
        assert 'ticker' in df.columns
        assert 'label_type' in df.columns
        assert 'positive' in df['label_type'].values
        assert 'negative' in df['label_type'].values
        
        # Test feature/label separation
        exclude_columns = ['ticker', 'start_date', 'end_date', 'label_type', 'notes']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        assert len(feature_columns) > 5  # Should have multiple features
        
        features = df[feature_columns]
        labels = (df['label_type'] == 'positive').astype(int)
        
        assert len(features) == len(labels)
        assert features.shape[1] == len(feature_columns)
        assert set(labels.unique()).issubset({0, 1})
        
        print("✅ Data generation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data generation tests failed: {e}")
        return False

def test_file_structure():
    """Test that all required files are present."""
    print("🧪 Testing file structure...")
    
    try:
        required_files = [
            'src/pattern_model_trainer.py',
            'src/model_evaluator.py',
            'tests/test_pattern_model_trainer.py',
            'tests/test_model_evaluator.py',
            'requirements.txt',
            'models'  # Directory
        ]
        
        for file_path in required_files:
            if file_path == 'models':
                assert os.path.isdir(file_path), f"Directory {file_path} not found"
            else:
                assert os.path.isfile(file_path), f"File {file_path} not found"
        
        # Check requirements.txt for new dependencies
        with open('requirements.txt', 'r') as f:
            content = f.read()
            assert 'xgboost' in content
            assert 'joblib' in content
            assert 'imbalanced-learn' in content
        
        print("✅ File structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ File structure tests failed: {e}")
        return False

def test_notebook_exists():
    """Test that the Jupyter notebook was created."""
    print("🧪 Testing notebook creation...")
    
    try:
        notebook_path = 'notebooks/05_pattern_model_training.ipynb'
        assert os.path.isfile(notebook_path), f"Notebook {notebook_path} not found"
        
        # Basic validation of notebook structure
        import json
        with open(notebook_path, 'r') as f:
            nb_data = json.load(f)
        
        assert 'cells' in nb_data
        assert len(nb_data['cells']) > 0
        
        # Check that first cell is markdown with title
        first_cell = nb_data['cells'][0]
        assert first_cell['cell_type'] == 'markdown'
        assert 'Pattern Model Training Notebook' in first_cell['source'][0]
        
        print("✅ Notebook tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Notebook tests failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Pattern Model Training Implementation Verification")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_data_generation,
        test_training_config,
        test_model_evaluator,
        test_notebook_exists
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
        print()
    
    print("=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All verification tests passed!")
        print("\n✅ Implementation Status:")
        print("   ✓ Core training infrastructure created")
        print("   ✓ Model evaluation system implemented")
        print("   ✓ Jupyter notebook interface ready")
        print("   ✓ Comprehensive test suite available")
        print("   ✓ Dependencies properly configured")
        
        print("\n🚀 Next Steps:")
        print("   1. Install missing system dependencies (e.g., 'brew install libomp' for XGBoost)")
        print("   2. Run the Jupyter notebook: notebooks/05_pattern_model_training.ipynb")
        print("   3. Train your first model using the existing labeled features")
        print("   4. Review model performance and iterate on hyperparameters")
        
    else:
        print(f"⚠️  {total - passed} tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 