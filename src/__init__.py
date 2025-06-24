# Stock Pattern Recognition Engine - Data Layer
__version__ = "0.1.0"

# Import existing modules
from .data_fetcher import (
    fetch_hk_stocks,
    validate_tickers,
    preview_cached_data,
    list_cached_tickers
)

# Note: BulkDataFetcher and HKStockUniverse modules not yet implemented

# Import new pattern labeling modules
from .pattern_labeler import (
    PatternLabel,
    PatternLabeler,
    LabelValidator,
    save_labeled_patterns,
    load_labeled_patterns,
    is_false_breakout,
    ValidationError,
    PatternLabelError
)

# Import signal outcome tagging modules
from .signal_outcome_tagger import (
    SignalOutcomeTagger,
    SignalOutcomeError,
    load_latest_matches,
    quick_tag_outcome,
    review_latest_feedback
)

# Import optional visualization (handle gracefully if not available)
try:
    from .pattern_visualizer import (
        PatternChartVisualizer,
        display_labeled_pattern,
        compare_patterns,
        VisualizationError
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

# Import feature extraction modules
from .feature_extractor import (
    FeatureExtractor,
    FeatureWindow,
    extract_features_from_labels,
    FeatureExtractionError
)

# Import technical indicators
from .technical_indicators import (
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

# Public API
__all__ = [
    # Data fetching
    "fetch_hk_stocks",
    "validate_tickers",
    "preview_cached_data",
    "list_cached_tickers",

    # Pattern labeling
    "PatternLabel",
    "PatternLabeler",
    "LabelValidator",
    "save_labeled_patterns",
    "load_labeled_patterns",
    "is_false_breakout",
    "ValidationError",
    "PatternLabelError",

    # Signal outcome tagging
    "SignalOutcomeTagger",
    "SignalOutcomeError",
    "load_latest_matches",
    "quick_tag_outcome",
    "review_latest_feedback",

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
]

# Add visualization exports if available
if _HAS_VISUALIZATION:
    __all__.extend([
        "PatternChartVisualizer",
        "display_labeled_pattern",
        "compare_patterns",
        "VisualizationError"
    ])
