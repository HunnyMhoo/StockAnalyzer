"""
Analysis subpackage for interactive analysis, data quality, and outcome tracking.
"""

from .interactive import (
    InteractivePatternAnalyzer,
    PatternAnalysisConfig,
    PatternAnalysisResult,
    SimplePatternConfig,
)

from .quality import (
    DataQualityAnalyzer,
    DataQualitySummary,
    show_enhanced_data_summary,
    quick_quality_check,
)

from .outcome import (
    SignalOutcomeTagger,
    SignalOutcomeError,
    load_latest_matches,
    quick_tag_outcome,
    review_latest_feedback,
)

from .evaluator import (
    ModelEvaluator,
    ModelEvaluationError,
    quick_evaluate_model,
)

# Create alias for backward compatibility
EvaluationResults = dict  # Results are returned as dictionaries

from .trainer import (
    PatternModelTrainer,
    TrainingConfig,
    TrainingResults,
    ModelTrainingError,
    load_trained_model,
    quick_train_model,
)

__all__ = [
    # Interactive analysis
    "InteractivePatternAnalyzer",
    "PatternAnalysisConfig",
    "PatternAnalysisResult",
    "SimplePatternConfig",
    
    # Data quality
    "DataQualityAnalyzer",
    "DataQualitySummary",
    "show_enhanced_data_summary",
    "quick_quality_check",
    
    # Signal outcome tracking
    "SignalOutcomeTagger",
    "SignalOutcomeError",
    "load_latest_matches",
    "quick_tag_outcome",
    "review_latest_feedback",
    
    # Model evaluation and training
    "ModelEvaluator",
    "ModelEvaluationError", 
    "EvaluationResults",
    "quick_evaluate_model",
    "PatternModelTrainer",
    "TrainingConfig",
    "TrainingResults",
    "ModelTrainingError",
    "load_trained_model",
    "quick_train_model",
] 