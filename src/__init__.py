# Stock Pattern Recognition Engine - Backward Compatibility Layer
"""
This module provides backward compatibility for existing code.
New code should import directly from stock_analyzer package.
"""

__version__ = "0.1.0"

# Re-export everything from the new package structure for backward compatibility
try:
    from stock_analyzer import *
    from stock_analyzer import __all__ as _stock_analyzer_all
    
    # Extend __all__ with the imported symbols
    __all__ = _stock_analyzer_all
    
except ImportError:
    # Fallback to old structure if new package not available
    import warnings
    warnings.warn(
        "Could not import from stock_analyzer package. "
        "Consider running: pip install -e . to install the package properly.",
        ImportWarning
    )
    
    # Import existing modules (original structure preserved for fallback)
    from .data_fetcher import (
        fetch_hk_stocks,
        validate_tickers,
        preview_cached_data,
        list_cached_tickers
    )

    # Import pattern labeling modules
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

    # Import interactive pattern analysis modules
    from .interactive_pattern_analyzer import (
        InteractivePatternAnalyzer,
        PatternAnalysisConfig,
        PatternAnalysisResult,
        SimplePatternConfig
    )

    # Import data quality analysis modules
    from .data_quality_analyzer import (
        DataQualityAnalyzer,
        DataQualitySummary,
        show_enhanced_data_summary,
        quick_quality_check
    )

    # Import notebook widget utilities (optional)
    try:
        from .notebook_widgets import (
            PatternAnalysisUI,
            WidgetConfig,
            create_pattern_analysis_interface,
            create_simple_input_form
        )
        _HAS_WIDGETS = True
    except ImportError:
        _HAS_WIDGETS = False

    # Fallback __all__ list
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

        # Interactive pattern analysis
        "InteractivePatternAnalyzer",
        "PatternAnalysisConfig",
        "PatternAnalysisResult",
        "SimplePatternConfig",

        # Data quality analysis
        "DataQualityAnalyzer",
        "DataQualitySummary",
        "show_enhanced_data_summary",
        "quick_quality_check",
    ]

    # Add widget exports if available
    if _HAS_WIDGETS:
        __all__.extend([
            "PatternAnalysisUI",
            "WidgetConfig", 
            "create_pattern_analysis_interface",
            "create_simple_input_form"
        ])

    # Add visualization exports if available
    if _HAS_VISUALIZATION:
        __all__.extend([
            "PatternChartVisualizer",
            "display_labeled_pattern",
            "compare_patterns",
            "VisualizationError"
        ])
