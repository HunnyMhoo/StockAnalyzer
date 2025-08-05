"""
Stock Analyzer - Hong Kong Stock Pattern Recognition System

A comprehensive package for analyzing Hong Kong stock patterns with technical indicators,
machine learning models, and interactive analysis tools.
"""

from typing import Any, Dict, List, Optional

__version__ = "0.1.0"
__author__ = "Stock Analyzer Team"
__email__ = "team@stockanalyzer.com"

# Import custom exceptions
# Core imports for the main package
from .analysis import (
    DataQualityAnalyzer,
    DataQualitySummary,
    InteractivePatternAnalyzer,
    ModelEvaluator,
    PatternAnalysisConfig,
    PatternAnalysisResult,
    PatternModelTrainer,
    SignalOutcomeError,
    SignalOutcomeTagger,
    SimplePatternConfig,
    load_latest_matches,
    quick_quality_check,
    quick_tag_outcome,
    review_latest_feedback,
    show_enhanced_data_summary,
)
from .data import (
    fetch_hk_stocks,
    list_cached_tickers,
    preview_cached_data,
    validate_tickers,
)
from .features import (
    FeatureExtractionError,
    FeatureExtractor,
    FeatureWindow,
    average_true_range,
    bollinger_bands,
    calculate_candle_patterns,
    calculate_drawdown_metrics,
    calculate_linear_trend_slope,
    detect_false_support_break,
    exponential_moving_average,
    extract_features_from_labels,
    find_recent_support_level,
    find_support_resistance_levels,
    macd,
    price_volatility,
    relative_strength_index,
    # Technical indicators
    simple_moving_average,
    volume_average_ratio,
)
from .patterns import (
    LabelValidator,
    PatternLabel,
    PatternLabeler,
    PatternLabelError,
    PatternScanner,
    PatternScanningError,
    ValidationError,
    is_false_breakout,
    load_labeled_patterns,
    save_labeled_patterns,
    scan_hk_stocks_for_patterns,
)
from .utils.errors import (
    DataFetchError,
    ModelTrainingError,
)

# Optional imports (gracefully handle missing dependencies)
try:
    from .visualization import (
        PatternChartVisualizer,
        VisualizationError,
        compare_patterns,
        display_labeled_pattern,
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
    "scan_hk_stocks_for_patterns",
    "PatternScanningError",
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
    # Public API functions
    "fetch_bulk",
    "build_features",
    "scan_patterns",
    "train_model",
]

# Add optional exports if available
if _HAS_VISUALIZATION:
    __all__.extend(
        [
            "PatternChartVisualizer",
            "display_labeled_pattern",
            "compare_patterns",
            "VisualizationError",
        ]
    )

if _HAS_WIDGETS:
    __all__.extend(
        [
            "PatternAnalysisUI",
            "WidgetConfig",
            "create_pattern_analysis_interface",
            "create_simple_input_form",
        ]
    )


# Public API functions for common tasks
def fetch_bulk(
    tickers: List[str],
    start_date: str,
    end_date: str,
    batch_size: int = 20,
    delay_between_batches: float = 2.0,
    max_retries: int = 2,
    force_refresh: bool = False,
    skip_failed: bool = True,
) -> Dict[str, Any]:
    """
    Fetch bulk stock data for multiple tickers.

    Args:
        tickers: List of stock ticker symbols (e.g., ['0700.HK', '0941.HK'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        batch_size: Number of stocks to process per batch
        delay_between_batches: Delay in seconds between batches
        max_retries: Number of retries for failed stocks
        force_refresh: If True, ignore cache and fetch fresh data
        skip_failed: If True, continue processing even if some stocks fail

    Returns:
        Dictionary containing fetch results and metadata

    Raises:
        DataFetchError: If bulk fetching fails
    """
    try:
        from .data import fetch_hk_stocks_bulk

        results = fetch_hk_stocks_bulk(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
            delay_between_batches=delay_between_batches,
            max_retries=max_retries,
            force_refresh=force_refresh,
            skip_failed=skip_failed,
        )
        return {
            "data": results,
            "tickers_requested": len(tickers),
            "tickers_fetched": len(results),
            "date_range": f"{start_date} to {end_date}",
        }
    except Exception as e:
        raise DataFetchError(f"Bulk data fetching failed: {e}") from e


def build_features(
    labeled_data_path: str,
    window_size: int = 30,
    prior_context_days: int = 30,
    support_lookback_days: int = 10,
    output_dir: str = "features",
) -> Dict[str, Any]:
    """
    Extract technical features from labeled pattern data.

    Args:
        labeled_data_path: Path to the labeled pattern data file
        window_size: Size of the feature extraction window
        prior_context_days: Number of days for prior context
        support_lookback_days: Number of days to look back for support levels
        output_dir: Directory to save extracted features

    Returns:
        Dictionary containing extracted features and metadata

    Raises:
        FeatureExtractionError: If feature extraction fails
    """
    try:
        from .features import FeatureExtractor

        extractor = FeatureExtractor(
            window_size=window_size,
            prior_context_days=prior_context_days,
            support_lookback_days=support_lookback_days,
            output_dir=output_dir,
        )

        features_df = extractor.extract_features_from_file(
            labels_file=labeled_data_path, save_to_file=True
        )

        return {
            "features": features_df,
            "feature_count": len(features_df.columns) if features_df is not None else 0,
            "sample_count": len(features_df) if features_df is not None else 0,
            "window_size": window_size,
            "output_dir": output_dir,
        }
    except Exception as e:
        raise FeatureExtractionError(f"Feature extraction failed: {e}") from e


def scan_patterns(
    tickers: List[str],
    model_path: str,
    min_confidence: float = 0.7,
    window_size: int = 30,
    max_windows_per_ticker: int = 10,
    save_results: bool = True,
    output_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Scan multiple stocks for trading patterns using a trained model.

    Args:
        tickers: List of stock ticker symbols to scan
        model_path: Path to the trained model file
        min_confidence: Minimum confidence for pattern detection
        window_size: Size of sliding window in days
        max_windows_per_ticker: Maximum number of windows to evaluate per ticker
        save_results: Whether to save results to file
        output_filename: Custom output filename (auto-generated if None)

    Returns:
        Dictionary containing scan results and metadata

    Raises:
        PatternScanningError: If pattern scanning fails
    """
    try:
        from .patterns import PatternScanner, ScanningConfig

        config = ScanningConfig(
            window_size=window_size,
            min_confidence=min_confidence,
            max_windows_per_ticker=max_windows_per_ticker,
            save_results=save_results,
            output_filename=output_filename,
        )

        scanner = PatternScanner(model_path=model_path)
        results = scanner.scan_tickers(tickers, config)

        return {
            "matches_df": results.matches_df,
            "scanning_summary": results.scanning_summary,
            "tickers_scanned": len(tickers),
            "min_confidence": min_confidence,
            "output_path": results.output_path,
        }
    except Exception as e:
        raise PatternScanningError(f"Pattern scanning failed: {e}") from e


def train_model(
    labeled_data_path: str,
    model_type: str = "xgboost",
    test_size: float = 0.2,
    use_cross_validation: bool = True,
    output_dir: str = "models",
) -> Dict[str, Any]:
    """
    Train a pattern detection model from labeled data.

    Args:
        labeled_data_path: Path to the labeled pattern data
        model_type: Type of model to train ('xgboost' or 'random_forest')
        test_size: Fraction of data to use for testing
        use_cross_validation: Whether to perform cross-validation
        output_dir: Directory to save the trained model

    Returns:
        Dictionary containing training results and model metadata

    Raises:
        ModelTrainingError: If model training fails
    """
    try:
        from .analysis import PatternModelTrainer, TrainingConfig

        config = TrainingConfig(
            model_type=model_type,
            test_size=test_size,
            use_cross_validation=use_cross_validation,
        )

        trainer = PatternModelTrainer(
            features_file=labeled_data_path, models_dir=output_dir, config=config
        )

        results = trainer.train()

        return {
            "model_path": results.model_path,
            "train_score": results.train_score,
            "test_score": results.test_score,
            "feature_importance": results.feature_importance,
            "model_type": model_type,
            "training_time": results.training_time,
        }
    except Exception as e:
        raise ModelTrainingError(f"Model training failed: {e}") from e
