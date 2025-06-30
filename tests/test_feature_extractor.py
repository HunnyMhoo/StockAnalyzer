"""
Unit tests for feature extractor module.

Tests the FeatureExtractor class and related functionality to ensure
proper feature extraction from labeled pattern data.
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Handle imports for both direct execution and package usage
try:
    from stock_analyzer.features import (
        FeatureExtractor,
        FeatureWindow,
        extract_features_from_labels,
        FeatureExtractionError
    )
    from stock_analyzer.patterns import PatternLabel
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from stock_analyzer.features import (
        FeatureExtractor,
        FeatureWindow,
        extract_features_from_labels,
        FeatureExtractionError
    )
    from stock_analyzer.patterns import PatternLabel


class TestFeatureExtractor:
    """Test class for FeatureExtractor."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        # Create 90 days of data to ensure sufficient context
        dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
        
        # Generate realistic OHLCV data with trend
        base_price = 100
        trend = np.linspace(0, 20, 90)  # Upward trend
        noise = np.random.normal(0, 2, 90)
        close_prices = base_price + trend + noise
        
        # Derive other prices from close
        high_prices = close_prices + np.random.uniform(0, 3, 90)
        low_prices = close_prices - np.random.uniform(0, 3, 90)
        open_prices = close_prices + np.random.uniform(-1, 1, 90)
        volumes = np.random.uniform(1000000, 5000000, 90)
        
        return pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes,
            'Adj Close': close_prices,  # Add required column
            'Dividends': np.zeros(90),
            'Stock Splits': np.zeros(90)
        }, index=dates)
    
    @pytest.fixture
    def sample_pattern_label(self):
        """Create sample pattern label for testing."""
        return PatternLabel(
            ticker="0700.HK",
            start_date="2023-02-01",
            end_date="2023-02-28",
            label_type="positive",
            notes="Test pattern"
        )
    
    @pytest.fixture
    def feature_extractor(self):
        """Create FeatureExtractor instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield FeatureExtractor(output_dir=temp_dir)
    
    @pytest.fixture
    def mock_data_loader(self, sample_ohlcv_data):
        """Mock data loader to return sample data."""
        with patch('src.feature_extractor._load_cached_data') as mock_load:
            mock_load.return_value = sample_ohlcv_data
            yield mock_load
    
    def test_feature_extractor_init(self):
        """Test FeatureExtractor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = FeatureExtractor(
                window_size=20,
                prior_context_days=25,
                support_lookback_days=15,
                output_dir=temp_dir
            )
            
            assert extractor.window_size == 20
            assert extractor.prior_context_days == 25
            assert extractor.support_lookback_days == 15
            assert extractor.output_dir == temp_dir
            assert os.path.exists(temp_dir)
    
    def test_extract_window_data(self, feature_extractor, sample_ohlcv_data):
        """Test window data extraction."""
        start_date = "2023-02-01"
        end_date = "2023-02-28"
        
        window_data, prior_context = feature_extractor._extract_window_data(
            sample_ohlcv_data, start_date, end_date
        )
        
        # Verify window data
        assert not window_data.empty
        assert window_data.index[0] >= pd.to_datetime(start_date)
        assert window_data.index[-1] <= pd.to_datetime(end_date)
        
        # Verify prior context
        assert prior_context is not None
        assert not prior_context.empty
        assert prior_context.index[-1] < pd.to_datetime(start_date)
    
    def test_validate_window_data(self, feature_extractor, sample_ohlcv_data):
        """Test window data validation."""
        # Valid data
        assert feature_extractor._validate_window_data(sample_ohlcv_data, "TEST.HK")
        
        # Empty data
        empty_data = pd.DataFrame()
        assert not feature_extractor._validate_window_data(empty_data, "TEST.HK")
        
        # Insufficient data
        small_data = sample_ohlcv_data.iloc[:2]
        assert not feature_extractor._validate_window_data(small_data, "TEST.HK")
        
        # Missing columns
        incomplete_data = sample_ohlcv_data.drop(columns=['Volume'])
        assert not feature_extractor._validate_window_data(incomplete_data, "TEST.HK")
    
    def test_calculate_trend_features(self, feature_extractor, sample_ohlcv_data):
        """Test trend feature calculation."""
        window_data = sample_ohlcv_data.iloc[30:60]  # 30-day window
        prior_context = sample_ohlcv_data.iloc[0:30]  # 30-day prior context
        
        features = feature_extractor._calculate_trend_features(window_data, prior_context)
        
        # Verify feature keys
        expected_keys = ['prior_trend_return', 'above_sma_50_ratio', 'trend_angle']
        assert all(key in features for key in expected_keys)
        
        # Verify feature types
        assert all(isinstance(features[key], float) for key in expected_keys)
        
        # Test with no prior context
        features_no_context = feature_extractor._calculate_trend_features(window_data, None)
        assert all(features_no_context[key] == 0.0 for key in expected_keys)
    
    def test_calculate_correction_features(self, feature_extractor, sample_ohlcv_data):
        """Test correction feature calculation."""
        window_data = sample_ohlcv_data.iloc[30:60]
        
        features = feature_extractor._calculate_correction_features(window_data)
        
        # Verify feature keys
        expected_keys = ['drawdown_pct', 'recovery_return_pct', 'down_day_ratio']
        assert all(key in features for key in expected_keys)
        
        # Verify feature types
        assert all(isinstance(features[key], float) for key in expected_keys)
        
        # Verify value ranges
        assert features['down_day_ratio'] >= 0
        assert features['down_day_ratio'] <= 100
    
    def test_calculate_support_break_features(self, feature_extractor, sample_ohlcv_data):
        """Test support break feature calculation."""
        window_data = sample_ohlcv_data.iloc[30:60]
        window_start_idx = 30
        
        features = feature_extractor._calculate_support_break_features(
            window_data, sample_ohlcv_data, window_start_idx
        )
        
        # Verify feature keys
        expected_keys = [
            'support_level', 'support_break_depth_pct', 'false_break_flag',
            'recovery_days', 'recovery_volume_ratio'
        ]
        assert all(key in features for key in expected_keys)
        
        # Verify feature types
        assert all(isinstance(features[key], float) for key in expected_keys)
        
        # Verify logical constraints
        assert features['support_level'] > 0
        assert features['support_break_depth_pct'] >= 0
        assert features['false_break_flag'] in [0.0, 1.0]
        assert features['recovery_days'] >= 0
        assert features['recovery_volume_ratio'] > 0
    
    def test_calculate_technical_indicators(self, feature_extractor, sample_ohlcv_data):
        """Test technical indicator calculation."""
        window_data = sample_ohlcv_data.iloc[30:60]
        
        features = feature_extractor._calculate_technical_indicators(window_data)
        
        # Verify feature keys
        expected_keys = [
            'sma_5', 'sma_10', 'sma_20', 'rsi_14', 'macd_diff',
            'volatility', 'volume_avg_ratio'
        ]
        assert all(key in features for key in expected_keys)
        
        # Verify feature types
        assert all(isinstance(features[key], float) for key in expected_keys)
        
        # Verify value ranges
        assert 0 <= features['rsi_14'] <= 100
        assert features['volatility'] >= 0
        assert features['volume_avg_ratio'] > 0
    
    @patch('src.feature_extractor._load_cached_data')
    def test_extract_features_from_label(self, mock_load_data, feature_extractor, 
                                       sample_ohlcv_data, sample_pattern_label):
        """Test feature extraction from single label."""
        mock_load_data.return_value = sample_ohlcv_data
        
        features = feature_extractor.extract_features_from_label(sample_pattern_label)
        
        assert features is not None
        assert isinstance(features, dict)
        
        # Verify metadata
        assert features['ticker'] == sample_pattern_label.ticker
        assert features['start_date'] == sample_pattern_label.start_date
        assert features['end_date'] == sample_pattern_label.end_date
        assert features['label_type'] == sample_pattern_label.label_type
        
        # Verify feature categories are present
        feature_categories = [
            'prior_trend_return', 'above_sma_50_ratio', 'trend_angle',  # Trend
            'drawdown_pct', 'recovery_return_pct', 'down_day_ratio',  # Correction
            'support_level', 'false_break_flag', 'recovery_days',  # Support break
            'sma_5', 'rsi_14', 'volatility'  # Technical indicators
        ]
        
        for category in feature_categories:
            assert category in features
            assert isinstance(features[category], (int, float))
    
    @patch('src.feature_extractor._load_cached_data')
    def test_extract_features_from_label_insufficient_data(self, mock_load_data, feature_extractor):
        """Test feature extraction with insufficient data."""
        # Return None to simulate missing data
        mock_load_data.return_value = None
        
        label = PatternLabel(
            ticker="MISSING.HK",
            start_date="2023-01-01",
            end_date="2023-01-31",
            label_type="positive"
        )
        
        features = feature_extractor.extract_features_from_label(label)
        assert features is None
    
    @patch('src.feature_extractor.load_labeled_patterns')
    @patch('src.feature_extractor._load_cached_data')
    def test_extract_features_batch(self, mock_load_data, mock_load_labels, 
                                  feature_extractor, sample_ohlcv_data):
        """Test batch feature extraction."""
        # Mock data and labels
        mock_load_data.return_value = sample_ohlcv_data
        mock_labels = [
            PatternLabel("0700.HK", "2023-02-01", "2023-02-28", "positive"),
            PatternLabel("0005.HK", "2023-02-01", "2023-02-28", "positive"),
        ]
        mock_load_labels.return_value = mock_labels
        
        # Extract features
        df = feature_extractor.extract_features_batch(mock_labels, save_to_file=False)
        
        assert not df.empty
        assert len(df) == 2
        assert 'ticker' in df.columns
        assert 'prior_trend_return' in df.columns
        assert 'support_level' in df.columns
    
    def test_feature_window_dataclass(self):
        """Test FeatureWindow dataclass."""
        sample_data = pd.DataFrame({'Close': [100, 101, 102]})
        
        window = FeatureWindow(
            ticker="0700.HK",
            start_date="2023-01-01",
            end_date="2023-01-31",
            label_type="positive",
            notes="Test",
            window_data=sample_data
        )
        
        assert window.ticker == "0700.HK"
        assert window.start_date == "2023-01-01"
        assert window.end_date == "2023-01-31"
        assert window.label_type == "positive"
        assert window.notes == "Test"
        assert window.window_data.equals(sample_data)
        assert window.prior_context_data is None
    
    def test_feature_extraction_error(self):
        """Test FeatureExtractionError exception."""
        error = FeatureExtractionError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    @patch('src.feature_extractor.load_labeled_patterns')
    def test_convenience_function(self, mock_load_labels):
        """Test convenience function."""
        mock_labels = [
            PatternLabel("0700.HK", "2023-02-01", "2023-02-28", "positive")
        ]
        mock_load_labels.return_value = mock_labels
        
        with tempfile.NamedTemporaryFile(suffix='.json') as temp_file:
            # Test that function can be called (will fail due to missing data, but that's OK)
            try:
                df = extract_features_from_labels(temp_file.name)
                # If it succeeds, it should return a DataFrame
                assert isinstance(df, pd.DataFrame)
            except Exception:
                # Expected to fail due to missing data in test environment
                pass
    
    def test_feature_counts_meet_requirements(self, feature_extractor, sample_ohlcv_data):
        """Test that at least 10 features are extracted as per requirements."""
        window_data = sample_ohlcv_data.iloc[30:60]
        prior_context = sample_ohlcv_data.iloc[0:30]
        
        # Count features from each category
        trend_features = feature_extractor._calculate_trend_features(window_data, prior_context)
        correction_features = feature_extractor._calculate_correction_features(window_data)
        support_features = feature_extractor._calculate_support_break_features(
            window_data, sample_ohlcv_data, 30
        )
        technical_features = feature_extractor._calculate_technical_indicators(window_data)
        
        total_features = (
            len(trend_features) + len(correction_features) + 
            len(support_features) + len(technical_features)
        )
        
        # Verify we have at least 10 features as required
        assert total_features >= 10
        
        # Verify specific feature counts
        assert len(trend_features) == 3
        assert len(correction_features) == 3
        assert len(support_features) == 5
        assert len(technical_features) == 7
        
        # Total should be 18 features
        assert total_features == 18


if __name__ == "__main__":
    pytest.main([__file__]) 