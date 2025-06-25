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
)

from .trainer import (
    PatternModelTrainer,
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
    "PatternModelTrainer",
] 