"""
Features subpackage for technical indicators and feature extraction.
"""

from .extractor import (
    FeatureExtractor,
    FeatureWindow,
    extract_features_from_labels,
    FeatureExtractionError,
)

from .indicators import (
    simple_moving_average,
    exponential_moving_average,
    relative_strength_index,
    macd,
    bollinger_bands,
    average_true_range,
    price_volatility,
    volume_average_ratio,
    find_support_resistance_levels,
    find_recent_support_level,
    calculate_linear_trend_slope,
    detect_false_support_break,
    calculate_drawdown_metrics,
    calculate_candle_patterns,
)

__all__ = [
    # Feature extraction
    "FeatureExtractor",
    "FeatureWindow",
    "extract_features_from_labels",
    "FeatureExtractionError",
    
    # Technical indicators
    "simple_moving_average",
    "exponential_moving_average",
    "relative_strength_index", 
    "macd",
    "bollinger_bands",
    "average_true_range",
    "price_volatility",
    "volume_average_ratio",
    "find_support_resistance_levels",
    "find_recent_support_level",
    "calculate_linear_trend_slope",
    "detect_false_support_break",
    "calculate_drawdown_metrics",
    "calculate_candle_patterns",
] 