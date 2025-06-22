"""
Unit tests for model evaluator module.

Tests the ModelEvaluator class and related functionality for comprehensive
model evaluation and visualization.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Handle imports for both direct execution and package usage
try:
    from src.model_evaluator import (
        ModelEvaluator,
        ModelEvaluationError,
        quick_evaluate_model
    )
    from src.pattern_model_trainer import TrainingResults, TrainingConfig
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.model_evaluator import (
        ModelEvaluator,
        ModelEvaluationError,
        quick_evaluate_model
    )
    from src.pattern_model_trainer import TrainingResults, TrainingConfig


class TestModelEvaluator:
    """Test class for ModelEvaluator."""
    
    @pytest.fixture
    def evaluator_with_temp_dirs(self):
        """Create evaluator instance with temporary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plots_dir = os.path.join(temp_dir, "plots")
            reports_dir = os.path.join(temp_dir, "reports")
            
            evaluator = ModelEvaluator(
                plots_dir=plots_dir,
                reports_dir=reports_dir,
                figsize=(8, 6),
                dpi=50  # Lower DPI for faster testing
            )
            
            yield evaluator, plots_dir, reports_dir, temp_dir
    
    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data for evaluation."""
        np.random.seed(42)
        n_samples = 100
        
        # Create balanced test data
        n_positive = 30
        n_negative = 70
        
        # Generate features
        X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.uniform(0, 10, n_samples),
            'feature3': np.random.exponential(1, n_samples),
            'feature4': np.random.beta(2, 5, n_samples)
        })
        
        # Generate labels
        y_test = pd.Series([1] * n_positive + [0] * n_negative)
        
        return X_test, y_test
    
    @pytest.fixture
    def mock_training_results(self):
        """Create mock training results for testing."""
        config = TrainingConfig(model_type="randomforest")
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4]])
        
        # Create mock feature importance
        feature_importance = pd.DataFrame({
            'feature': ['feature1', 'feature2', 'feature3', 'feature4'],
            'importance': [0.1, 0.4, 0.3, 0.2]
        }).sort_values('importance', ascending=False)
        
        results = TrainingResults(
            model=mock_model,
            scaler=None,
            config=config,
            train_score={'accuracy': 0.95},
            test_score={'accuracy': 0.90},
            cv_scores={'cv_accuracy_mean': 0.92},
            feature_importance=feature_importance,
            model_path="/test/model.pkl",
            training_time=10.0,
            metadata={'samples': 100}
        )
        
        return results
    
    def test_evaluator_initialization(self, evaluator_with_temp_dirs):
        """Test ModelEvaluator initialization."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        assert evaluator.plots_dir == plots_dir
        assert evaluator.reports_dir == reports_dir
        assert evaluator.figsize == (8, 6)
        assert evaluator.dpi == 50
        
        # Check that directories were created
        assert os.path.exists(plots_dir)
        assert os.path.exists(reports_dir)
    
    def test_calculate_detailed_metrics(self, evaluator_with_temp_dirs):
        """Test detailed metrics calculation."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        # Create sample predictions
        y_test = pd.Series([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])  # One false negative
        y_pred_proba = np.array([0.8, 0.2, 0.4, 0.1, 0.9])
        
        metrics = evaluator._calculate_detailed_metrics(y_test, y_pred, y_pred_proba)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'balanced_accuracy', 'precision', 'recall', 
            'f1_score', 'matthews_corrcoef', 'roc_auc', 'average_precision', 'log_loss'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
        
        # Check specific values
        assert metrics['accuracy'] == 0.8  # 4/5 correct
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['average_precision'] <= 1
    
    def test_analyze_confusion_matrix(self, evaluator_with_temp_dirs):
        """Test confusion matrix analysis."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        # Create sample predictions with known confusion matrix
        y_test = pd.Series([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1])  # TP=2, TN=2, FP=1, FN=1
        
        analysis = evaluator._analyze_confusion_matrix(y_test, y_pred)
        
        # Check confusion matrix components
        assert 'confusion_matrix' in analysis
        assert analysis['true_positives'] == 2
        assert analysis['true_negatives'] == 2
        assert analysis['false_positives'] == 1
        assert analysis['false_negatives'] == 1
        
        # Check derived metrics
        assert analysis['specificity'] == 2/3  # TN/(TN+FP) = 2/(2+1)
        assert analysis['sensitivity'] == 2/3  # TP/(TP+FN) = 2/(2+1)
        assert analysis['false_positive_rate'] == 1/3  # FP/(FP+TN)
        assert analysis['false_negative_rate'] == 1/3  # FN/(FN+TP)
    
    def test_analyze_prediction_confidence(self, evaluator_with_temp_dirs):
        """Test prediction confidence analysis."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        y_test = pd.Series([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])  # One incorrect prediction
        y_pred_proba = np.array([0.9, 0.2, 0.4, 0.1])
        
        analysis = evaluator._analyze_prediction_confidence(y_test, y_pred, y_pred_proba)
        
        # Check confidence statistics
        assert 'mean_confidence' in analysis
        assert 'std_confidence' in analysis
        assert 'min_confidence' in analysis
        assert 'max_confidence' in analysis
        
        assert analysis['mean_confidence'] == np.mean(y_pred_proba)
        assert analysis['min_confidence'] == 0.1
        assert analysis['max_confidence'] == 0.9
        
        # Check confidence by correctness
        assert 'mean_confidence_correct' in analysis
        assert 'mean_confidence_incorrect' in analysis
        
        # Confidence bins
        assert 'bin_accuracy' in analysis
        assert 'bin_counts' in analysis
        assert 'confidence_bins' in analysis
        assert len(analysis['bin_accuracy']) == 10  # 10 bins
    
    def test_analyze_misclassifications(self, evaluator_with_temp_dirs, sample_test_data):
        """Test misclassification analysis."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        X_test, y_test = sample_test_data
        
        # Create predictions with some misclassifications
        y_pred = y_test.copy().values
        y_pred[0] = 1 - y_pred[0]  # Flip first prediction
        y_pred[1] = 1 - y_pred[1]  # Flip second prediction
        
        y_pred_proba = np.random.uniform(0.1, 0.9, len(y_test))
        
        analysis = evaluator._analyze_misclassifications(X_test, y_test, y_pred, y_pred_proba)
        
        # Check analysis results
        assert analysis['num_misclassified'] == 2
        assert len(analysis['misclassified_samples']) == 2
        
        # Check that misclassified samples have required columns
        misc_df = analysis['misclassified_samples']
        required_columns = ['true_label', 'predicted_label', 'confidence', 'error_type']
        for col in required_columns:
            assert col in misc_df.columns
        
        # Check false positives and negatives
        assert 'false_positives' in analysis
        assert 'false_negatives' in analysis
        assert analysis['num_false_positives'] + analysis['num_false_negatives'] == 2
    
    def test_analyze_misclassifications_no_errors(self, evaluator_with_temp_dirs, sample_test_data):
        """Test misclassification analysis with perfect predictions."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        X_test, y_test = sample_test_data
        
        # Perfect predictions
        y_pred = y_test.values
        y_pred_proba = np.random.uniform(0.1, 0.9, len(y_test))
        
        analysis = evaluator._analyze_misclassifications(X_test, y_test, y_pred, y_pred_proba)
        
        assert analysis['num_misclassified'] == 0
        assert len(analysis['misclassified_samples']) == 0
        assert len(analysis['false_positives']) == 0
        assert len(analysis['false_negatives']) == 0
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_confusion_matrix(self, mock_close, mock_savefig, evaluator_with_temp_dirs):
        """Test confusion matrix plot generation."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        y_test = pd.Series([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        
        plot_path = evaluator._plot_confusion_matrix(y_test, y_pred, "test_model")
        
        # Check that plot was saved
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        assert plot_path is not None
        assert "confusion_matrix_test_model.png" in plot_path
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_roc_curve(self, mock_close, mock_savefig, evaluator_with_temp_dirs):
        """Test ROC curve plot generation."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        y_test = pd.Series([1, 0, 1, 0, 1, 0])
        y_pred_proba = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1])
        
        plot_path = evaluator._plot_roc_curve(y_test, y_pred_proba, "test_model")
        
        # Check that plot was saved
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        assert plot_path is not None
        assert "roc_curve_test_model.png" in plot_path
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_feature_importance(self, mock_close, mock_savefig, evaluator_with_temp_dirs):
        """Test feature importance plot generation."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        feature_importance = pd.DataFrame({
            'feature': ['feature1', 'feature2', 'feature3'],
            'importance': [0.5, 0.3, 0.2]
        })
        
        plot_path = evaluator._plot_feature_importance(feature_importance, "test_model")
        
        # Check that plot was saved
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        assert plot_path is not None
        assert "feature_importance_test_model.png" in plot_path
    
    def test_save_evaluation_report(self, evaluator_with_temp_dirs):
        """Test evaluation report saving."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        # Create mock evaluation results
        evaluation_results = {
            'metrics': {
                'accuracy': 0.9,
                'precision': 0.85,
                'recall': 0.88,
                'f1_score': 0.86
            },
            'confusion_matrix': {
                'confusion_matrix': np.array([[10, 2], [1, 7]]),
                'true_positives': 7,
                'true_negatives': 10,
                'false_positives': 2,
                'false_negatives': 1,
                'specificity': 0.83,
                'sensitivity': 0.88
            },
            'confidence_analysis': {
                'mean_confidence': 0.75,
                'std_confidence': 0.2,
                'mean_confidence_correct': 0.8,
                'mean_confidence_incorrect': 0.6
            },
            'misclassification_analysis': {
                'num_misclassified': 3,
                'num_false_positives': 2,
                'num_false_negatives': 1
            },
            'plot_paths': ['/path/to/plot1.png', '/path/to/plot2.png']
        }
        
        report_path = evaluator._save_evaluation_report(evaluation_results, "test_model")
        
        # Check that report file was created
        assert os.path.exists(report_path)
        assert "evaluation_report_test_model.txt" in report_path
        
        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
            
        assert "Model Evaluation Report - test_model" in content
        assert "PERFORMANCE METRICS" in content
        assert "accuracy: 0.9000" in content
        assert "CONFUSION MATRIX ANALYSIS" in content
        assert "CONFIDENCE ANALYSIS" in content
        assert "MISCLASSIFICATION ANALYSIS" in content
    
    def test_evaluate_model_from_results(self, evaluator_with_temp_dirs, mock_training_results, sample_test_data):
        """Test model evaluation from TrainingResults."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        X_test, y_test = sample_test_data
        
        # Limit test data to match mock predictions
        X_test_small = X_test.iloc[:4]
        y_test_small = y_test.iloc[:4]
        
        with patch.object(evaluator, '_generate_all_plots', return_value=[]):
            eval_results = evaluator.evaluate_model_from_results(
                results=mock_training_results,
                X_test=X_test_small,
                y_test=y_test_small,
                generate_plots=False,
                save_report=False
            )
        
        # Check evaluation results structure
        assert 'metrics' in eval_results
        assert 'confusion_matrix' in eval_results
        assert 'confidence_analysis' in eval_results
        assert 'misclassification_analysis' in eval_results
        
        # Check that model predictions were used
        mock_training_results.model.predict.assert_called_once()
        mock_training_results.model.predict_proba.assert_called_once()
    
    def test_compare_models(self, evaluator_with_temp_dirs):
        """Test model comparison functionality."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        # Create mock evaluation results for two models
        model1_results = {
            'metrics': {
                'accuracy': 0.9,
                'precision': 0.85,
                'recall': 0.88,
                'f1_score': 0.86,
                'roc_auc': 0.92
            }
        }
        
        model2_results = {
            'metrics': {
                'accuracy': 0.88,
                'precision': 0.90,
                'recall': 0.82,
                'f1_score': 0.86,
                'roc_auc': 0.89
            }
        }
        
        model_results = [
            ("model1", model1_results),
            ("model2", model2_results)
        ]
        
        with patch.object(evaluator, '_plot_model_comparison', return_value="/path/to/comparison.png"):
            comparison_results = evaluator.compare_models(model_results, save_comparison=False)
        
        # Check comparison results structure
        assert 'comparison_table' in comparison_results
        assert 'best_models' in comparison_results
        assert 'summary' in comparison_results
        
        # Check comparison table
        comp_table = comparison_results['comparison_table']
        assert len(comp_table) == 2
        assert 'model_name' in comp_table.columns
        assert 'accuracy' in comp_table.columns
        
        # Check best models
        best_models = comparison_results['best_models']
        assert 'accuracy' in best_models
        assert best_models['accuracy']['model'] == 'model1'  # Higher accuracy
        assert best_models['precision']['model'] == 'model2'  # Higher precision
    
    def test_compare_models_insufficient_models(self, evaluator_with_temp_dirs):
        """Test model comparison with insufficient models."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        model_results = [("model1", {'metrics': {'accuracy': 0.9}})]
        
        with pytest.raises(ModelEvaluationError, match="Need at least 2 models"):
            evaluator.compare_models(model_results)
    
    def test_generate_comparison_summary(self, evaluator_with_temp_dirs):
        """Test comparison summary generation."""
        evaluator, plots_dir, reports_dir, temp_dir = evaluator_with_temp_dirs
        
        comparison_df = pd.DataFrame({
            'model_name': ['model1', 'model2'],
            'accuracy': [0.9, 0.88],
            'f1_score': [0.86, 0.84],
            'precision': [0.85, 0.90]
        })
        
        best_models = {
            'accuracy': {'model': 'model1', 'value': 0.9},
            'f1_score': {'model': 'model1', 'value': 0.86},
            'precision': {'model': 'model2', 'value': 0.90}
        }
        
        summary = evaluator._generate_comparison_summary(comparison_df, best_models)
        
        assert "MODEL COMPARISON SUMMARY" in summary
        assert "Best Overall Model (F1-Score): model1" in summary
        assert "Best Models by Metric:" in summary
        assert "accuracy: model1" in summary
        assert "precision: model2" in summary
        assert "Performance Ranges:" in summary


class TestModelEvaluationFromFile:
    """Test class for model evaluation from saved files."""
    
    @pytest.fixture
    def mock_model_package(self):
        """Create mock model package for file loading tests."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4]])
        mock_model.feature_importances_ = np.array([0.1, 0.4, 0.3, 0.2])
        
        return {
            'model': mock_model,
            'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
            'scaler': None,
            'config': TrainingConfig(model_type="randomforest"),
            'metadata': {'training_date': '2023-01-01'}
        }
    
    def test_evaluate_model_from_file_success(self, mock_model_package):
        """Test successful model evaluation from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(plots_dir=temp_dir, reports_dir=temp_dir)
            
            # Create test data
            X_test = pd.DataFrame({
                'feature1': [1, 2, 3, 4],
                'feature2': [0.1, 0.2, 0.3, 0.4],
                'feature3': [10, 20, 30, 40],
                'feature4': [0.5, 0.6, 0.7, 0.8]
            })
            y_test = pd.Series([1, 0, 1, 0])
            
            with patch('src.model_evaluator.load_trained_model', return_value=mock_model_package):
                with patch.object(evaluator, '_generate_all_plots', return_value=[]):
                    eval_results = evaluator.evaluate_model_from_file(
                        model_path="test_model.pkl",
                        X_test=X_test,
                        y_test=y_test,
                        generate_plots=False,
                        save_report=False
                    )
            
            # Check evaluation results
            assert 'metrics' in eval_results
            assert 'feature_importance' in eval_results
            
            # Check that model was called with correct features
            mock_model_package['model'].predict.assert_called_once()
            mock_model_package['model'].predict_proba.assert_called_once()
    
    def test_evaluate_model_from_file_missing_features(self, mock_model_package):
        """Test model evaluation with missing features."""
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(plots_dir=temp_dir, reports_dir=temp_dir)
            
            # Create test data missing some features
            X_test = pd.DataFrame({
                'feature1': [1, 2, 3, 4],
                'feature2': [0.1, 0.2, 0.3, 0.4]
                # Missing feature3 and feature4
            })
            y_test = pd.Series([1, 0, 1, 0])
            
            with patch('src.model_evaluator.load_trained_model', return_value=mock_model_package):
                with pytest.raises(ModelEvaluationError, match="Missing features"):
                    evaluator.evaluate_model_from_file(
                        model_path="test_model.pkl",
                        X_test=X_test,
                        y_test=y_test
                    )
    
    def test_evaluate_model_from_file_with_scaler(self, mock_model_package):
        """Test model evaluation with feature scaling."""
        # Add scaler to model package
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [-1, -1, -1, -1], [0.5, 0.5, 0.5, 0.5]])
        mock_model_package['scaler'] = mock_scaler
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(plots_dir=temp_dir, reports_dir=temp_dir)
            
            X_test = pd.DataFrame({
                'feature1': [1, 2, 3, 4],
                'feature2': [0.1, 0.2, 0.3, 0.4],
                'feature3': [10, 20, 30, 40],
                'feature4': [0.5, 0.6, 0.7, 0.8]
            })
            y_test = pd.Series([1, 0, 1, 0])
            
            with patch('src.model_evaluator.load_trained_model', return_value=mock_model_package):
                with patch.object(evaluator, '_generate_all_plots', return_value=[]):
                    eval_results = evaluator.evaluate_model_from_file(
                        model_path="test_model.pkl",
                        X_test=X_test,
                        y_test=y_test,
                        generate_plots=False,
                        save_report=False
                    )
            
            # Check that scaler was applied
            mock_scaler.transform.assert_called_once()
            
            # Check evaluation results
            assert 'metrics' in eval_results


class TestQuickEvaluateModel:
    """Test class for quick_evaluate_model convenience function."""
    
    @patch('src.model_evaluator.ModelEvaluator')
    def test_quick_evaluate_model(self, mock_evaluator_class):
        """Test quick evaluation function."""
        # Mock evaluator instance and evaluation results
        mock_evaluator = MagicMock()
        mock_results = {'metrics': {'accuracy': 0.9}}
        mock_evaluator.evaluate_model_from_file.return_value = mock_results
        mock_evaluator_class.return_value = mock_evaluator
        
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = pd.Series([1, 0, 1])
        
        result = quick_evaluate_model(
            model_path="test_model.pkl",
            X_test=X_test,
            y_test=y_test,
            plots_dir="custom_plots"
        )
        
        # Check that evaluator was created with correct parameters
        mock_evaluator_class.assert_called_once()
        call_args = mock_evaluator_class.call_args[1]
        assert call_args['plots_dir'] == "custom_plots"
        
        # Check that evaluation was called
        mock_evaluator.evaluate_model_from_file.assert_called_once_with(
            "test_model.pkl", X_test, y_test
        )
        assert result == mock_results


# Integration tests
class TestEvaluatorIntegration:
    """Integration tests for the model evaluator."""
    
    def test_end_to_end_evaluation(self):
        """Test complete evaluation pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(
                plots_dir=temp_dir,
                reports_dir=temp_dir
            )
            
            # Create synthetic data
            np.random.seed(42)
            X_test = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 50),
                'feature2': np.random.uniform(0, 1, 50)
            })
            y_test = pd.Series(np.random.choice([0, 1], 50))
            
            # Create mock model with realistic behavior
            mock_model = MagicMock()
            
            # Generate somewhat realistic predictions
            predictions = []
            probabilities = []
            for i, (_, row) in enumerate(X_test.iterrows()):
                # Simple linear combination for prediction
                score = 0.3 * row['feature1'] + 0.7 * row['feature2']
                prob = 1 / (1 + np.exp(-score))  # Sigmoid
                pred = 1 if prob > 0.5 else 0
                
                predictions.append(pred)
                probabilities.append([1-prob, prob])
            
            mock_model.predict.return_value = np.array(predictions)
            mock_model.predict_proba.return_value = np.array(probabilities)
            mock_model.feature_importances_ = np.array([0.3, 0.7])
            
            # Create training results
            config = TrainingConfig(model_type="randomforest")
            feature_importance = pd.DataFrame({
                'feature': ['feature2', 'feature1'],
                'importance': [0.7, 0.3]
            })
            
            results = TrainingResults(
                model=mock_model,
                scaler=None,
                config=config,
                train_score={'accuracy': 0.95},
                test_score={'accuracy': 0.90},
                cv_scores={},
                feature_importance=feature_importance,
                model_path="/test/model.pkl",
                training_time=10.0,
                metadata={}
            )
            
            # This should complete without errors
            with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                eval_results = evaluator.evaluate_model_from_results(
                    results=results,
                    X_test=X_test,
                    y_test=y_test,
                    generate_plots=True,
                    save_report=True
                )
            
            # Check that all components were executed
            assert 'metrics' in eval_results
            assert 'confusion_matrix' in eval_results
            assert 'confidence_analysis' in eval_results
            assert 'misclassification_analysis' in eval_results
            assert 'feature_importance' in eval_results
            
            # Check that plots were "generated" (mocked)
            assert 'plot_paths' in eval_results
            
            # Check that report was saved
            assert 'report_path' in eval_results
            assert os.path.exists(eval_results['report_path'])


if __name__ == "__main__":
    pytest.main([__file__])