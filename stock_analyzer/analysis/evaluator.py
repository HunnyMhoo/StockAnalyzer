"""
Model Evaluator for Pattern Detection Models

This module provides comprehensive evaluation and visualization capabilities
for trained pattern detection models including performance metrics, plots,
and misclassification analysis.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Handle imports for model loading
try:
    from .pattern_model_trainer import load_trained_model, TrainingResults
except ImportError:
    from pattern_model_trainer import load_trained_model, TrainingResults


# Configuration constants
PLOTS_DIR = "plots"
REPORTS_DIR = "reports"
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 100


class ModelEvaluationError(Exception):
    """Custom exception for model evaluation errors."""
    pass


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization class.
    
    This class provides tools for:
    - Performance metric calculation and reporting
    - Visualization of model performance (ROC, precision-recall, confusion matrix)
    - Feature importance analysis
    - Misclassification analysis and confidence scoring
    - Comparative analysis between multiple models
    """
    
    def __init__(self, 
                 plots_dir: str = PLOTS_DIR,
                 reports_dir: str = REPORTS_DIR,
                 figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
                 dpi: int = DEFAULT_DPI):
        """
        Initialize ModelEvaluator.
        
        Args:
            plots_dir: Directory for saving plots
            reports_dir: Directory for saving evaluation reports
            figsize: Default figure size for plots
            dpi: Resolution for saved plots
        """
        self.plots_dir = plots_dir
        self.reports_dir = reports_dir
        self.figsize = figsize
        self.dpi = dpi
        self._ensure_directories()
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        for directory in [self.plots_dir, self.reports_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"✓ Created directory: {directory}/")
    
    def evaluate_model_from_results(self, 
                                   results: TrainingResults,
                                   X_test: pd.DataFrame,
                                   y_test: pd.Series,
                                   generate_plots: bool = True,
                                   save_report: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a trained model from TrainingResults.
        
        Args:
            results: TrainingResults object from training
            X_test: Test features
            y_test: Test labels
            generate_plots: Whether to generate visualization plots
            save_report: Whether to save evaluation report
            
        Returns:
            Dictionary containing all evaluation metrics and artifacts
        """
        model = results.model
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        return self._comprehensive_evaluation(
            model, X_test, y_test, y_pred, y_pred_proba,
            feature_importance=results.feature_importance,
            model_name=f"{results.config.model_type}_model",
            generate_plots=generate_plots,
            save_report=save_report
        )
    
    def evaluate_model_from_file(self,
                                model_path: str,
                                X_test: pd.DataFrame,
                                y_test: pd.Series,
                                generate_plots: bool = True,
                                save_report: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a model loaded from file.
        
        Args:
            model_path: Path to saved model file
            X_test: Test features
            y_test: Test labels
            generate_plots: Whether to generate visualization plots
            save_report: Whether to save evaluation report
            
        Returns:
            Dictionary containing all evaluation metrics and artifacts
        """
        try:
            # Load model
            model_package = load_trained_model(model_path)
            model = model_package['model']
            
            # Validate features
            expected_features = model_package['feature_names']
            if not all(feat in X_test.columns for feat in expected_features):
                missing_features = [f for f in expected_features if f not in X_test.columns]
                raise ModelEvaluationError(f"Missing features: {missing_features}")
            
            # Reorder features to match training order
            X_test_ordered = X_test[expected_features]
            
            # Apply scaling if used during training
            if model_package.get('scaler') is not None:
                scaler = model_package['scaler']
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test_ordered),
                    columns=X_test_ordered.columns,
                    index=X_test_ordered.index
                )
            else:
                X_test_scaled = X_test_ordered
            
            # Generate predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Extract feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': expected_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            model_name = f"{model_package['config'].model_type}_loaded"
            
            return self._comprehensive_evaluation(
                model, X_test_scaled, y_test, y_pred, y_pred_proba,
                feature_importance=feature_importance,
                model_name=model_name,
                generate_plots=generate_plots,
                save_report=save_report
            )
            
        except Exception as e:
            if isinstance(e, ModelEvaluationError):
                raise
            raise ModelEvaluationError(f"Model evaluation failed: {e}")
    
    def _comprehensive_evaluation(self,
                                 model: Any,
                                 X_test: pd.DataFrame,
                                 y_test: pd.Series,
                                 y_pred: np.ndarray,
                                 y_pred_proba: np.ndarray,
                                 feature_importance: Optional[pd.DataFrame] = None,
                                 model_name: str = "model",
                                 generate_plots: bool = True,
                                 save_report: bool = True) -> Dict[str, Any]:
        """
        Internal method for comprehensive model evaluation.
        """
        evaluation_results = {}
        
        print(f"📊 Evaluating {model_name}...")
        
        # Basic metrics
        evaluation_results['metrics'] = self._calculate_detailed_metrics(y_test, y_pred, y_pred_proba)
        
        # Confusion matrix analysis
        evaluation_results['confusion_matrix'] = self._analyze_confusion_matrix(y_test, y_pred)
        
        # Prediction confidence analysis
        evaluation_results['confidence_analysis'] = self._analyze_prediction_confidence(
            y_test, y_pred, y_pred_proba
        )
        
        # Misclassification analysis
        evaluation_results['misclassification_analysis'] = self._analyze_misclassifications(
            X_test, y_test, y_pred, y_pred_proba
        )
        
        # Feature importance (if available)
        if feature_importance is not None:
            evaluation_results['feature_importance'] = feature_importance
        
        # Generate plots
        plot_paths = []
        if generate_plots:
            plot_paths = self._generate_all_plots(
                y_test, y_pred, y_pred_proba, feature_importance, model_name
            )
        evaluation_results['plot_paths'] = plot_paths
        
        # Save comprehensive report
        if save_report:
            report_path = self._save_evaluation_report(evaluation_results, model_name)
            evaluation_results['report_path'] = report_path
        
        print("✅ Model evaluation completed!")
        return evaluation_results
    
    def _calculate_detailed_metrics(self, 
                                   y_test: pd.Series, 
                                   y_pred: np.ndarray, 
                                   y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss, matthews_corrcoef, balanced_accuracy_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
        }
        
        # ROC AUC (only if both classes present)
        if len(np.unique(y_test)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
            metrics['log_loss'] = log_loss(y_test, y_pred_proba)
        else:
            metrics['roc_auc'] = 0.0
            metrics['average_precision'] = 0.0
            metrics['log_loss'] = 0.0
        
        print("📈 Detailed Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return metrics
    
    def _analyze_confusion_matrix(self, 
                                 y_test: pd.Series, 
                                 y_pred: np.ndarray) -> Dict[str, Union[np.ndarray, int, float]]:
        """Analyze confusion matrix and derived statistics."""
        cm = confusion_matrix(y_test, y_pred)
        
        # Extract confusion matrix components
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            analysis = {
                'confusion_matrix': cm,
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            }
        else:
            analysis = {
                'confusion_matrix': cm,
                'specificity': 0.0,
                'sensitivity': 0.0,
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0
            }
        
        return analysis
    
    def _analyze_prediction_confidence(self, 
                                      y_test: pd.Series, 
                                      y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence and reliability."""
        confidence_analysis = {
            'mean_confidence': float(np.mean(y_pred_proba)),
            'std_confidence': float(np.std(y_pred_proba)),
            'min_confidence': float(np.min(y_pred_proba)),
            'max_confidence': float(np.max(y_pred_proba))
        }
        
        # Analyze confidence by correctness
        correct_mask = (y_test.values == y_pred)
        
        if np.any(correct_mask):
            confidence_analysis['mean_confidence_correct'] = float(np.mean(y_pred_proba[correct_mask]))
        else:
            confidence_analysis['mean_confidence_correct'] = 0.0
            
        if np.any(~correct_mask):
            confidence_analysis['mean_confidence_incorrect'] = float(np.mean(y_pred_proba[~correct_mask]))
        else:
            confidence_analysis['mean_confidence_incorrect'] = 0.0
        
        # Confidence bins
        confidence_bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        bin_indices = np.digitize(y_pred_proba, confidence_bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(confidence_bins) - 2)
        
        bin_accuracy = []
        bin_counts = []
        for i in range(len(confidence_bins) - 1):
            bin_mask = (bin_indices == i)
            if np.any(bin_mask):
                bin_acc = np.mean(y_test.values[bin_mask] == y_pred[bin_mask])
                bin_accuracy.append(float(bin_acc))
                bin_counts.append(int(np.sum(bin_mask)))
            else:
                bin_accuracy.append(0.0)
                bin_counts.append(0)
        
        confidence_analysis['bin_accuracy'] = bin_accuracy
        confidence_analysis['bin_counts'] = bin_counts
        confidence_analysis['confidence_bins'] = confidence_bins.tolist()
        
        return confidence_analysis
    
    def _analyze_misclassifications(self, 
                                   X_test: pd.DataFrame,
                                   y_test: pd.Series, 
                                   y_pred: np.ndarray, 
                                   y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze misclassified samples for insights."""
        # Find misclassified samples
        misclassified_mask = (y_test.values != y_pred)
        
        if not np.any(misclassified_mask):
            return {
                'num_misclassified': 0,
                'misclassified_samples': pd.DataFrame(),
                'false_positives': pd.DataFrame(),
                'false_negatives': pd.DataFrame()
            }
        
        # Create misclassification dataframe
        misclassified_df = X_test[misclassified_mask].copy()
        misclassified_df['true_label'] = y_test.values[misclassified_mask]
        misclassified_df['predicted_label'] = y_pred[misclassified_mask]
        misclassified_df['confidence'] = y_pred_proba[misclassified_mask]
        misclassified_df['error_type'] = ['FP' if true == 0 else 'FN' 
                                         for true in misclassified_df['true_label']]
        
        # Separate false positives and false negatives
        false_positives = misclassified_df[misclassified_df['error_type'] == 'FP'].copy()
        false_negatives = misclassified_df[misclassified_df['error_type'] == 'FN'].copy()
        
        analysis = {
            'num_misclassified': int(np.sum(misclassified_mask)),
            'misclassified_samples': misclassified_df,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'num_false_positives': len(false_positives),
            'num_false_negatives': len(false_negatives)
        }
        
        print(f"🔍 Misclassification Analysis:")
        print(f"   Total misclassified: {analysis['num_misclassified']}")
        print(f"   False positives: {analysis['num_false_positives']}")
        print(f"   False negatives: {analysis['num_false_negatives']}")
        
        return analysis
    
    def _generate_all_plots(self, 
                           y_test: pd.Series, 
                           y_pred: np.ndarray, 
                           y_pred_proba: np.ndarray,
                           feature_importance: Optional[pd.DataFrame],
                           model_name: str) -> List[str]:
        """Generate all visualization plots."""
        plot_paths = []
        
        try:
            # Confusion Matrix
            cm_path = self._plot_confusion_matrix(y_test, y_pred, model_name)
            if cm_path:
                plot_paths.append(cm_path)
            
            # ROC Curve
            if len(np.unique(y_test)) > 1:
                roc_path = self._plot_roc_curve(y_test, y_pred_proba, model_name)
                if roc_path:
                    plot_paths.append(roc_path)
                
                # Precision-Recall Curve
                pr_path = self._plot_precision_recall_curve(y_test, y_pred_proba, model_name)
                if pr_path:
                    plot_paths.append(pr_path)
            
            # Feature Importance
            if feature_importance is not None:
                fi_path = self._plot_feature_importance(feature_importance, model_name)
                if fi_path:
                    plot_paths.append(fi_path)
            
            # Prediction Distribution
            dist_path = self._plot_prediction_distribution(y_test, y_pred_proba, model_name)
            if dist_path:
                plot_paths.append(dist_path)
            
        except Exception as e:
            print(f"⚠ Error generating plots: {e}")
        
        return plot_paths
    
    def _plot_confusion_matrix(self, 
                              y_test: pd.Series, 
                              y_pred: np.ndarray, 
                              model_name: str) -> Optional[str]:
        """Generate confusion matrix heatmap."""
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, f'confusion_matrix_{model_name}.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            
            print(f"✓ Saved confusion matrix: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"⚠ Failed to generate confusion matrix: {e}")
            plt.close()
            return None
    
    def _plot_roc_curve(self, 
                       y_test: pd.Series, 
                       y_pred_proba: np.ndarray, 
                       model_name: str) -> Optional[str]:
        """Generate ROC curve plot."""
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, f'roc_curve_{model_name}.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            
            print(f"✓ Saved ROC curve: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"⚠ Failed to generate ROC curve: {e}")
            plt.close()
            return None
    
    def _plot_precision_recall_curve(self, 
                                    y_test: pd.Series, 
                                    y_pred_proba: np.ndarray, 
                                    model_name: str) -> Optional[str]:
        """Generate Precision-Recall curve plot."""
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            # Calculate Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Plot PR curve
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AP = {avg_precision:.3f})')
            
            # Plot baseline
            baseline = np.sum(y_test) / len(y_test)
            plt.axhline(y=baseline, color='red', linestyle='--', alpha=0.5,
                       label=f'Baseline (AP = {baseline:.3f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, f'precision_recall_{model_name}.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            
            print(f"✓ Saved PR curve: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"⚠ Failed to generate PR curve: {e}")
            plt.close()
            return None
    
    def _plot_feature_importance(self, 
                                feature_importance: pd.DataFrame, 
                                model_name: str) -> Optional[str]:
        """Generate feature importance bar plot."""
        try:
            plt.figure(figsize=(12, 8), dpi=self.dpi)
            
            # Take top 15 features
            top_features = feature_importance.head(15)
            
            # Create horizontal bar plot
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importance - {model_name}')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(top_features['importance']):
                plt.text(v + 0.001, i, f'{v:.4f}', va='center')
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, f'feature_importance_{model_name}.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            
            print(f"✓ Saved feature importance: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"⚠ Failed to generate feature importance plot: {e}")
            plt.close()
            return None
    
    def _plot_prediction_distribution(self, 
                                     y_test: pd.Series, 
                                     y_pred_proba: np.ndarray, 
                                     model_name: str) -> Optional[str]:
        """Generate prediction probability distribution plot."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
            
            # Separate probabilities by true class
            pos_probs = y_pred_proba[y_test == 1]
            neg_probs = y_pred_proba[y_test == 0]
            
            # Plot 1: Histogram of probabilities by class
            ax1.hist(neg_probs, bins=20, alpha=0.7, label='True Negative', color='red')
            ax1.hist(pos_probs, bins=20, alpha=0.7, label='True Positive', color='blue')
            ax1.set_xlabel('Predicted Probability')
            ax1.set_ylabel('Count')
            ax1.set_title('Prediction Probability Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Box plot of probabilities by class
            box_data = [neg_probs, pos_probs] if len(neg_probs) > 0 and len(pos_probs) > 0 else [y_pred_proba]
            box_labels = ['True Negative', 'True Positive'] if len(neg_probs) > 0 and len(pos_probs) > 0 else ['All']
            
            ax2.boxplot(box_data, labels=box_labels)
            ax2.set_ylabel('Predicted Probability')
            ax2.set_title('Probability Distribution by True Class')
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'Prediction Analysis - {model_name}')
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, f'prediction_distribution_{model_name}.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            
            print(f"✓ Saved prediction distribution: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"⚠ Failed to generate prediction distribution plot: {e}")
            plt.close()
            return None
    
    def _save_evaluation_report(self, 
                               evaluation_results: Dict[str, Any], 
                               model_name: str) -> str:
        """Save comprehensive evaluation report to file."""
        try:
            report_path = os.path.join(self.reports_dir, f'evaluation_report_{model_name}.txt')
            
            with open(report_path, 'w') as f:
                f.write(f"Model Evaluation Report - {model_name}\n")
                f.write("=" * 50 + "\n\n")
                
                # Metrics section
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 20 + "\n")
                for metric, value in evaluation_results['metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
                
                # Confusion Matrix section
                f.write("CONFUSION MATRIX ANALYSIS\n")
                f.write("-" * 25 + "\n")
                cm_analysis = evaluation_results['confusion_matrix']
                f.write(f"Confusion Matrix:\n{cm_analysis['confusion_matrix']}\n")
                if 'true_positives' in cm_analysis:
                    f.write(f"True Positives: {cm_analysis['true_positives']}\n")
                    f.write(f"True Negatives: {cm_analysis['true_negatives']}\n")
                    f.write(f"False Positives: {cm_analysis['false_positives']}\n")
                    f.write(f"False Negatives: {cm_analysis['false_negatives']}\n")
                    f.write(f"Specificity: {cm_analysis['specificity']:.4f}\n")
                    f.write(f"Sensitivity: {cm_analysis['sensitivity']:.4f}\n")
                f.write("\n")
                
                # Confidence Analysis
                f.write("CONFIDENCE ANALYSIS\n")
                f.write("-" * 19 + "\n")
                conf_analysis = evaluation_results['confidence_analysis']
                f.write(f"Mean Confidence: {conf_analysis['mean_confidence']:.4f}\n")
                f.write(f"Std Confidence: {conf_analysis['std_confidence']:.4f}\n")
                f.write(f"Mean Confidence (Correct): {conf_analysis['mean_confidence_correct']:.4f}\n")
                f.write(f"Mean Confidence (Incorrect): {conf_analysis['mean_confidence_incorrect']:.4f}\n")
                f.write("\n")
                
                # Misclassification Analysis
                f.write("MISCLASSIFICATION ANALYSIS\n")
                f.write("-" * 27 + "\n")
                misc_analysis = evaluation_results['misclassification_analysis']
                f.write(f"Total Misclassified: {misc_analysis['num_misclassified']}\n")
                f.write(f"False Positives: {misc_analysis['num_false_positives']}\n")
                f.write(f"False Negatives: {misc_analysis['num_false_negatives']}\n")
                f.write("\n")
                
                # Feature Importance (if available)
                if 'feature_importance' in evaluation_results:
                    f.write("TOP 10 FEATURE IMPORTANCE\n")
                    f.write("-" * 26 + "\n")
                    top_features = evaluation_results['feature_importance'].head(10)
                    for _, row in top_features.iterrows():
                        f.write(f"{row['feature']}: {row['importance']:.4f}\n")
                    f.write("\n")
                
                # Generated Files
                f.write("GENERATED FILES\n")
                f.write("-" * 15 + "\n")
                for plot_path in evaluation_results['plot_paths']:
                    f.write(f"Plot: {plot_path}\n")
            
            print(f"✓ Saved evaluation report: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"⚠ Failed to save evaluation report: {e}")
            return ""
    
    def compare_models(self, 
                      model_results: List[Tuple[str, Dict[str, Any]]],
                      save_comparison: bool = True) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: List of tuples (model_name, evaluation_results)
            save_comparison: Whether to save comparison report
            
        Returns:
            Dictionary containing comparison analysis
        """
        if len(model_results) < 2:
            raise ModelEvaluationError("Need at least 2 models for comparison")
        
        print(f"📊 Comparing {len(model_results)} models...")
        
        # Extract metrics for comparison
        comparison_data = []
        for model_name, results in model_results:
            metrics = results['metrics'].copy()
            metrics['model_name'] = model_name
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best model for each metric
        metric_columns = [col for col in comparison_df.columns if col != 'model_name']
        best_models = {}
        
        for metric in metric_columns:
            best_idx = comparison_df[metric].idxmax()
            best_models[metric] = {
                'model': comparison_df.loc[best_idx, 'model_name'],
                'value': comparison_df.loc[best_idx, metric]
            }
        
        comparison_results = {
            'comparison_table': comparison_df,
            'best_models': best_models,
            'summary': self._generate_comparison_summary(comparison_df, best_models)
        }
        
        # Generate comparison plot
        if len(model_results) <= 5:  # Only plot if reasonable number of models
            try:
                plot_path = self._plot_model_comparison(comparison_df)
                comparison_results['comparison_plot'] = plot_path
            except Exception as e:
                print(f"⚠ Failed to generate comparison plot: {e}")
        
        # Save comparison report
        if save_comparison:
            report_path = self._save_comparison_report(comparison_results)
            comparison_results['report_path'] = report_path
        
        print("✅ Model comparison completed!")
        return comparison_results
    
    def _generate_comparison_summary(self, 
                                    comparison_df: pd.DataFrame, 
                                    best_models: Dict[str, Dict]) -> str:
        """Generate text summary of model comparison."""
        summary_lines = []
        summary_lines.append("MODEL COMPARISON SUMMARY")
        summary_lines.append("=" * 25)
        summary_lines.append("")
        
        # Overall best model (by F1 score)
        if 'f1_score' in best_models:
            best_overall = best_models['f1_score']['model']
            best_f1 = best_models['f1_score']['value']
            summary_lines.append(f"Best Overall Model (F1-Score): {best_overall} ({best_f1:.4f})")
            summary_lines.append("")
        
        # Best models by category
        summary_lines.append("Best Models by Metric:")
        for metric, info in best_models.items():
            summary_lines.append(f"  {metric}: {info['model']} ({info['value']:.4f})")
        
        summary_lines.append("")
        
        # Performance ranges
        summary_lines.append("Performance Ranges:")
        for col in comparison_df.columns:
            if col != 'model_name':
                min_val = comparison_df[col].min()
                max_val = comparison_df[col].max()
                summary_lines.append(f"  {col}: {min_val:.4f} - {max_val:.4f}")
        
        return "\n".join(summary_lines)
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame) -> Optional[str]:
        """Generate model comparison radar/bar chart."""
        try:
            # Select key metrics for visualization
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            available_metrics = [m for m in key_metrics if m in comparison_df.columns]
            
            if len(available_metrics) < 3:
                return None
            
            plt.figure(figsize=(12, 8), dpi=self.dpi)
            
            # Create grouped bar chart
            x = np.arange(len(available_metrics))
            width = 0.8 / len(comparison_df)
            
            for i, (_, row) in enumerate(comparison_df.iterrows()):
                values = [row[metric] for metric in available_metrics]
                plt.bar(x + i * width, values, width, label=row['model_name'], alpha=0.8)
            
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x + width * (len(comparison_df) - 1) / 2, available_metrics, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, 'model_comparison.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=self.dpi)
            plt.close()
            
            print(f"✓ Saved comparison plot: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"⚠ Failed to generate comparison plot: {e}")
            plt.close()
            return None
    
    def _save_comparison_report(self, comparison_results: Dict[str, Any]) -> str:
        """Save model comparison report."""
        try:
            report_path = os.path.join(self.reports_dir, 'model_comparison_report.txt')
            
            with open(report_path, 'w') as f:
                f.write("MODEL COMPARISON REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Summary
                f.write(comparison_results['summary'])
                f.write("\n\n")
                
                # Detailed comparison table
                f.write("DETAILED COMPARISON TABLE\n")
                f.write("-" * 25 + "\n")
                f.write(comparison_results['comparison_table'].to_string())
                f.write("\n\n")
                
                # Generated files
                if 'comparison_plot' in comparison_results:
                    f.write("GENERATED FILES\n")
                    f.write("-" * 15 + "\n")
                    f.write(f"Comparison Plot: {comparison_results['comparison_plot']}\n")
            
            print(f"✓ Saved comparison report: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"⚠ Failed to save comparison report: {e}")
            return ""


def quick_evaluate_model(model_path: str,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        **kwargs) -> Dict[str, Any]:
    """
    Convenience function for quick model evaluation.
    
    Args:
        model_path: Path to saved model
        X_test: Test features
        y_test: Test labels
        **kwargs: Additional evaluator parameters
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = ModelEvaluator(**kwargs)
    return evaluator.evaluate_model_from_file(model_path, X_test, y_test)