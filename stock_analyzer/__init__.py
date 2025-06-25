"""
Stock Analyzer - Hong Kong Stock Pattern Recognition System

A comprehensive package for analyzing Hong Kong stock patterns with technical indicators,
machine learning models, and interactive analysis tools.
"""

__version__ = "0.1.0"
__author__ = "Stock Analyzer Team"
__email__ = "team@stockanalyzer.com"

# Core imports for the main package
from .data import (
    fetch_hk_stocks,
    validate_tickers,
    preview_cached_data,
    list_cached_tickers,
)

from .features import (
    FeatureExtractor,
    FeatureWindow,
    extract_features_from_labels,
    FeatureExtractionError,
    # Technical indicators
    simple_moving_average,
    relative_strength_index,
    macd,
    price_volatility,
    volume_average_ratio,
    find_recent_support_level,
    calculate_linear_trend_slope,
    detect_false_support_break,
    calculate_drawdown_metrics,
)

from .patterns import (
    PatternLabel,
    PatternLabeler,
    LabelValidator,
    save_labeled_patterns,
    load_labeled_patterns,
    is_false_breakout,
    ValidationError,
    PatternLabelError,
    PatternScanner,
    scan_patterns,
    PatternScannerError,
)

from .analysis import (
    InteractivePatternAnalyzer,
    PatternAnalysisConfig,
    PatternAnalysisResult,
    SimplePatternConfig,
    DataQualityAnalyzer,
    DataQualitySummary,
    show_enhanced_data_summary,
    quick_quality_check,
    SignalOutcomeTagger,
    SignalOutcomeError,
    load_latest_matches,
    quick_tag_outcome,
    review_latest_feedback,
    ModelEvaluator,
    PatternModelTrainer,
)

# Optional imports (gracefully handle missing dependencies)
try:
    from .visualization import (
        PatternChartVisualizer,
        display_labeled_pattern,
        compare_patterns,
        VisualizationError,
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

try:
    from .utils import (
        PatternAnalysisUI,
        WidgetConfig,
        create_pattern_analysis_interface,
        create_simple_input_form,
    )
    _HAS_WIDGETS = True
except ImportError:
    _HAS_WIDGETS = False

# Public API
__all__ = [
    # Version info
    "__version__",
    
    # Data fetching
    "fetch_hk_stocks",
    "validate_tickers", 
    "preview_cached_data",
    "list_cached_tickers",
    
    # Feature extraction
    "FeatureExtractor",
    "FeatureWindow",
    "extract_features_from_labels",
    "FeatureExtractionError",
    
    # Technical indicators
    "simple_moving_average",
    "relative_strength_index",
    "macd",
    "price_volatility",
    "volume_average_ratio",
    "find_recent_support_level",
    "calculate_linear_trend_slope",
    "detect_false_support_break",
    "calculate_drawdown_metrics",
    
    # Pattern analysis
    "PatternLabel",
    "PatternLabeler",
    "LabelValidator",
    "save_labeled_patterns",
    "load_labeled_patterns",
    "is_false_breakout",
    "ValidationError",
    "PatternLabelError",
    "PatternScanner",
    "scan_patterns",
    "PatternScannerError",
    
    # Analysis tools
    "InteractivePatternAnalyzer",
    "PatternAnalysisConfig",
    "PatternAnalysisResult",
    "SimplePatternConfig",
    "DataQualityAnalyzer",
    "DataQualitySummary",
    "show_enhanced_data_summary",
    "quick_quality_check",
    "SignalOutcomeTagger",
    "SignalOutcomeError",
    "load_latest_matches",
    "quick_tag_outcome",
    "review_latest_feedback",
    "ModelEvaluator",
    "PatternModelTrainer",
]

# Add optional exports if available
if _HAS_VISUALIZATION:
    __all__.extend([
        "PatternChartVisualizer",
        "display_labeled_pattern",
        "compare_patterns",
        "VisualizationError",
    ])

if _HAS_WIDGETS:
    __all__.extend([
        "PatternAnalysisUI",
        "WidgetConfig",
        "create_pattern_analysis_interface",
        "create_simple_input_form",
    ]) 