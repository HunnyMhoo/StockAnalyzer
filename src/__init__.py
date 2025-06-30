# Stock Pattern Recognition Engine - Backward Compatibility Layer
"""
This module provides backward compatibility for existing code.
New code should import directly from stock_analyzer package.

DEPRECATED: This src module is deprecated. Use stock_analyzer package instead.
"""

import warnings
__version__ = "0.1.0"

# Issue deprecation warning
warnings.warn(
    "The 'src' module is deprecated. Please use 'stock_analyzer' package instead. "
    "Example: 'from stock_analyzer.data import fetch_hk_stocks' instead of 'from src.data_fetcher import fetch_hk_stocks'",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new package structure for backward compatibility
try:
    # Data fetching and management
    from stock_analyzer.data import (
        fetch_hk_stocks,
        validate_tickers,
        preview_cached_data,
        list_cached_tickers,
        fetch_hk_stocks_bulk,
        fetch_all_major_hk_stocks,
        create_bulk_fetch_summary,
        get_hk_stock_list_static,
        get_top_hk_stocks,
        get_hk_stocks_by_sector,
        get_comprehensive_hk_stock_list,
        validate_hk_tickers_batch,
        BulkCollectionConfig,
        BulkCollector,
        ResultsManager,
        ProgressTracker,
        BulkCollectionError,
        create_beginner_collector,
        create_enterprise_collector,
        quick_demo,
    )

    # Feature extraction and technical indicators
    from stock_analyzer.features import (
        FeatureExtractor,
        FeatureWindow,
        extract_features_from_labels,
        FeatureExtractionError,
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

    # Pattern detection and labeling
    from stock_analyzer.patterns import (
        PatternLabel,
        PatternLabeler,
        LabelValidator,
        save_labeled_patterns,
        load_labeled_patterns,
        is_false_breakout,
        ValidationError,
        PatternLabelError,
        PatternScanner,
        ScanningConfig,
        scan_hk_stocks_for_patterns,
        PatternScanningError,
    )

    # Analysis and training
    from stock_analyzer.analysis import (
        PatternModelTrainer,
        TrainingConfig,
        TrainingResults,
        ModelTrainingError,
        load_trained_model,
        quick_train_model,
        ModelEvaluator,
        EvaluationResults,
        ModelEvaluationError,
        SignalOutcomeTagger,
        SignalOutcomeError,
        load_latest_matches,
        quick_tag_outcome,
        review_latest_feedback,
        DataQualityAnalyzer,
        DataQualitySummary,
        show_enhanced_data_summary,
        quick_quality_check,
        InteractivePatternAnalyzer,
        PatternAnalysisConfig,
        PatternAnalysisResult,
        SimplePatternConfig,
    )

    # Visualization (optional)
    try:
        from stock_analyzer.visualization import (
            PatternChartVisualizer,
            display_labeled_pattern,
            compare_patterns,
            VisualizationError,
        )
        _HAS_VISUALIZATION = True
    except ImportError:
        _HAS_VISUALIZATION = False

    # Utilities (optional)
    try:
        from stock_analyzer.utils import (
            PatternAnalysisUI,
            WidgetConfig,
            create_pattern_analysis_interface,
            create_simple_input_form,
        )
        _HAS_WIDGETS = True
    except ImportError:
        _HAS_WIDGETS = False

    # Define what's available for import
    __all__ = [
        # Data fetching
        "fetch_hk_stocks",
        "validate_tickers",
        "preview_cached_data",
        "list_cached_tickers",
        "fetch_hk_stocks_bulk",
        "fetch_all_major_hk_stocks",
        "create_bulk_fetch_summary",
        
        # HK stock universe
        "get_hk_stock_list_static",
        "get_top_hk_stocks",
        "get_hk_stocks_by_sector",
        "get_comprehensive_hk_stock_list",
        "validate_hk_tickers_batch",
        
        # Bulk collection
        "BulkCollectionConfig",
        "BulkCollector",
        "ResultsManager",
        "ProgressTracker",
        "BulkCollectionError",
        "create_beginner_collector",
        "create_enterprise_collector",
        "quick_demo",
        
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
        
        # Pattern labeling
        "PatternLabel",
        "PatternLabeler",
        "LabelValidator",
        "save_labeled_patterns",
        "load_labeled_patterns",
        "is_false_breakout",
        "ValidationError",
        "PatternLabelError",
        
        # Pattern scanning
        "PatternScanner",
        "ScanningConfig",
        "scan_hk_stocks_for_patterns",
        "PatternScanningError",
        
        # Model training
        "PatternModelTrainer",
        "TrainingConfig",
        "TrainingResults",
        "ModelTrainingError",
        "load_trained_model",
        "quick_train_model",
        
        # Model evaluation
        "ModelEvaluator",
        "EvaluationResults",
        "ModelEvaluationError",
        
        # Signal outcome tagging
        "SignalOutcomeTagger",
        "SignalOutcomeError",
        "load_latest_matches",
        "quick_tag_outcome",
        "review_latest_feedback",
        
        # Data quality analysis
        "DataQualityAnalyzer",
        "DataQualitySummary",
        "show_enhanced_data_summary",
        "quick_quality_check",
        
        # Interactive analysis
        "InteractivePatternAnalyzer",
        "PatternAnalysisConfig",
        "PatternAnalysisResult",
        "SimplePatternConfig",
    ]
    
    # Add optional components to __all__ if available
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

except ImportError as e:
    # Fallback error message if new package not available
    raise ImportError(
        f"Could not import from stock_analyzer package: {e}\n"
        "Please ensure the stock_analyzer package is properly installed.\n"
        "Run: pip install -e . from the project root directory."
    ) from e
