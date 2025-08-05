"""
Custom exceptions for the stock analyzer package.

This module provides basic custom exceptions that can be used throughout
the package for consistent error handling.
"""


class StockAnalyzerError(Exception):
    """Base exception for all stock analyzer errors."""

    pass


class DataFetchError(StockAnalyzerError):
    """Raised when data fetching operations fail."""

    pass


class FeatureExtractionError(StockAnalyzerError):
    """Raised when feature extraction operations fail."""

    pass


class PatternScanningError(StockAnalyzerError):
    """Raised when pattern scanning operations fail."""

    pass


class ModelTrainingError(StockAnalyzerError):
    """Raised when model training operations fail."""

    pass


class ValidationError(StockAnalyzerError):
    """Raised when data validation fails."""

    pass


class ConfigurationError(StockAnalyzerError):
    """Raised when configuration is invalid."""

    pass
