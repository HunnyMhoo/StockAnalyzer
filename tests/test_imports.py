"""
Test module for verifying that all imports work correctly in the new package structure.
"""

import pytest
from importlib import import_module


def test_main_package_import():
    """Test that the main stock_analyzer package can be imported."""
    import stock_analyzer
    assert hasattr(stock_analyzer, '__version__')
    assert stock_analyzer.__version__ == "0.1.0"


def test_data_subpackage_imports():
    """Test that data subpackage modules can be imported."""
    # Test subpackage import
    from stock_analyzer.data import fetch_hk_stocks, validate_tickers
    
    # Test main package import (should work via re-exports)
    from stock_analyzer import fetch_hk_stocks as main_fetch_hk_stocks
    
    # They should be the same function
    assert fetch_hk_stocks is main_fetch_hk_stocks


def test_features_subpackage_imports():
    """Test that features subpackage modules can be imported."""
    # Test subpackage import
    from stock_analyzer.features import FeatureExtractor, simple_moving_average
    
    # Test main package import
    from stock_analyzer import FeatureExtractor as main_FeatureExtractor
    
    assert FeatureExtractor is main_FeatureExtractor


def test_patterns_subpackage_imports():
    """Test that patterns subpackage modules can be imported."""
    # Test subpackage import
    from stock_analyzer.patterns import PatternLabeler, PatternScanner
    
    # Test main package import
    from stock_analyzer import PatternLabeler as main_PatternLabeler
    
    assert PatternLabeler is main_PatternLabeler


def test_analysis_subpackage_imports():
    """Test that analysis subpackage modules can be imported."""
    # Test subpackage import
    from stock_analyzer.analysis import InteractivePatternAnalyzer, DataQualityAnalyzer
    
    # Test main package import
    from stock_analyzer import InteractivePatternAnalyzer as main_InteractivePatternAnalyzer
    
    assert InteractivePatternAnalyzer is main_InteractivePatternAnalyzer


def test_optional_imports():
    """Test that optional imports work gracefully."""
    try:
        from stock_analyzer.visualization import PatternChartVisualizer
        from stock_analyzer import PatternChartVisualizer as main_PatternChartVisualizer
        assert PatternChartVisualizer is main_PatternChartVisualizer
    except ImportError:
        # Should be gracefully handled
        pass
    
    try:
        from stock_analyzer.utils import PatternAnalysisUI
        from stock_analyzer import PatternAnalysisUI as main_PatternAnalysisUI
        assert PatternAnalysisUI is main_PatternAnalysisUI
    except ImportError:
        # Should be gracefully handled
        pass


def test_backward_compatibility():
    """Test that the old import structure still works."""
    # Test that old imports still work
    import src
    assert hasattr(src, '__version__')
    
    # Test specific imports from src
    from src import fetch_hk_stocks, FeatureExtractor, PatternLabeler
    
    # Verify they are the same as new imports
    from stock_analyzer import (
        fetch_hk_stocks as new_fetch_hk_stocks,
        FeatureExtractor as new_FeatureExtractor,
        PatternLabeler as new_PatternLabeler
    )
    
    # They should be the same functions/classes (due to re-export)
    assert fetch_hk_stocks is new_fetch_hk_stocks
    assert FeatureExtractor is new_FeatureExtractor
    assert PatternLabeler is new_PatternLabeler


def test_all_main_exports():
    """Test that __all__ exports are properly defined and importable."""
    import stock_analyzer
    
    # Test that __all__ is defined
    assert hasattr(stock_analyzer, '__all__')
    assert isinstance(stock_analyzer.__all__, list)
    assert len(stock_analyzer.__all__) > 0
    
    # Test that all exports in __all__ are actually available
    for export_name in stock_analyzer.__all__:
        assert hasattr(stock_analyzer, export_name), f"Export '{export_name}' not found in stock_analyzer"


def test_subpackage_structure():
    """Test that the subpackage structure is correctly set up."""
    # Test that subpackages can be imported
    subpackages = ['data', 'features', 'patterns', 'analysis']
    
    for subpackage in subpackages:
        module = import_module(f'stock_analyzer.{subpackage}')
        assert hasattr(module, '__all__')
        assert isinstance(module.__all__, list)
    
    # Test optional subpackages
    optional_subpackages = ['visualization', 'utils']
    for subpackage in optional_subpackages:
        try:
            module = import_module(f'stock_analyzer.{subpackage}')
            assert hasattr(module, '__all__')
            assert isinstance(module.__all__, list)
        except ImportError:
            # Optional subpackages may not be available
            pass


if __name__ == "__main__":
    pytest.main([__file__]) 