"""
Tests for Pattern Match Visualizer

This module tests the pattern match visualization functionality including:
- Match data loading and validation
- Single match visualization 
- Batch processing capabilities
- Chart generation and saving
- Error handling scenarios
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.append('../src')

from pattern_visualizer import (
    PatternChartVisualizer,
    MatchVisualizationError,
    MatchRow,
    visualize_match,
    visualize_matches_from_csv,
    plot_match,
    analyze_matches_by_confidence,
    generate_matches_report
)


class TestMatchDataHandling:
    """Test match data loading and validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = PatternChartVisualizer(charts_dir=self.temp_dir)
        
        # Sample matches data
        self.sample_matches = pd.DataFrame([
            {
                'ticker': '0700.HK',
                'window_start_date': '2023-10-01',
                'window_end_date': '2023-10-31',
                'confidence_score': 0.85,
                'rank': 1
            },
            {
                'ticker': '0005.HK',
                'window_start_date': '2023-09-15',
                'window_end_date': '2023-10-15',
                'confidence_score': 0.78,
                'rank': 2
            },
            {
                'ticker': '0388.HK',
                'window_start_date': '2023-11-01',
                'window_end_date': '2023-11-30',
                'confidence_score': 0.72,
                'rank': 3
            }
        ])
    
    def test_load_matches_from_csv_valid_file(self):
        """Test loading valid matches CSV file."""
        # Create test CSV file
        csv_path = os.path.join(self.temp_dir, 'test_matches.csv')
        self.sample_matches.to_csv(csv_path, index=False)
        
        # Load matches
        loaded_matches = self.visualizer.load_matches_from_csv(csv_path)
        
        assert len(loaded_matches) == 3
        assert 'ticker' in loaded_matches.columns
        assert 'confidence_score' in loaded_matches.columns
        assert loaded_matches.iloc[0]['ticker'] == '0700.HK'
    
    def test_load_matches_from_csv_missing_file(self):
        """Test error handling for missing CSV file."""
        non_existent_path = os.path.join(self.temp_dir, 'missing.csv')
        
        with pytest.raises(MatchVisualizationError, match="Matches file not found"):
            self.visualizer.load_matches_from_csv(non_existent_path)
    
    def test_load_matches_from_csv_missing_columns(self):
        """Test error handling for CSV with missing required columns."""
        # Create CSV with missing columns
        invalid_data = pd.DataFrame([
            {'ticker': '0700.HK', 'confidence_score': 0.8}  # Missing date columns
        ])
        
        csv_path = os.path.join(self.temp_dir, 'invalid_matches.csv')
        invalid_data.to_csv(csv_path, index=False)
        
        with pytest.raises(MatchVisualizationError, match="Missing required columns"):
            self.visualizer.load_matches_from_csv(csv_path)
    
    def test_validate_match_data_valid(self):
        """Test validation of valid match data."""
        result = self.visualizer.validate_match_data(self.sample_matches)
        assert result is True
    
    def test_validate_match_data_null_values(self):
        """Test validation error for null values in critical columns."""
        invalid_matches = self.sample_matches.copy()
        invalid_matches.loc[0, 'ticker'] = None
        
        with pytest.raises(MatchVisualizationError, match="Found null values"):
            self.visualizer.validate_match_data(invalid_matches)
    
    def test_validate_match_data_invalid_date_range(self):
        """Test validation error for invalid date ranges."""
        invalid_matches = self.sample_matches.copy()
        invalid_matches.loc[0, 'window_start_date'] = '2023-10-31'
        invalid_matches.loc[0, 'window_end_date'] = '2023-10-01'  # End before start
        
        with pytest.raises(MatchVisualizationError, match="Invalid date range"):
            self.visualizer.validate_match_data(invalid_matches)
    
    def test_match_row_dataclass(self):
        """Test MatchRow dataclass functionality."""
        match_row = MatchRow(
            ticker='0700.HK',
            window_start_date='2023-10-01',
            window_end_date='2023-10-31',
            confidence_score=0.85,
            rank=1,
            support_level=150.0
        )
        
        assert match_row.ticker == '0700.HK'
        assert match_row.confidence_score == 0.85
        assert match_row.support_level == 150.0


class TestSupportLevelCalculation:
    """Test support level calculation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = PatternChartVisualizer(charts_dir=self.temp_dir)
        
        # Create sample OHLCV data
        dates = pd.date_range('2023-10-01', periods=30, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 30),
            'High': np.random.uniform(150, 250, 30),
            'Low': np.random.uniform(90, 150, 30),
            'Close': np.random.uniform(100, 200, 30),
            'Volume': np.random.uniform(1000000, 5000000, 30)
        }, index=dates)
    
    @patch('pattern_visualizer.find_support_resistance_levels')
    def test_calculate_support_level_with_data(self, mock_support_levels):
        """Test support level calculation with valid data."""
        # Mock the support/resistance function
        mock_support_levels.return_value = (
            pd.Series([100.0, 105.0, 110.0], index=self.sample_data.index[:3]),
            pd.Series([200.0, 205.0, 210.0], index=self.sample_data.index[:3])
        )
        
        support_level = self.visualizer._calculate_support_level(
            self.sample_data,
            '2023-10-01',
            '2023-10-15'
        )
        
        assert isinstance(support_level, float)
        assert support_level > 0
        mock_support_levels.assert_called_once()
    
    def test_calculate_support_level_no_window_data(self):
        """Test support level calculation with no data in window."""
        support_level = self.visualizer._calculate_support_level(
            self.sample_data,
            '2024-01-01',  # Future date with no data
            '2024-01-15'
        )
        
        # Should fall back to overall minimum
        assert isinstance(support_level, float)
        assert support_level == self.sample_data['Low'].min()
    
    @patch('pattern_visualizer.find_support_resistance_levels')
    def test_calculate_support_level_with_error(self, mock_support_levels):
        """Test support level calculation error handling."""
        mock_support_levels.side_effect = Exception("Calculation error")
        
        # Should not raise exception, should return fallback value
        support_level = self.visualizer._calculate_support_level(
            self.sample_data,
            '2023-10-01',
            '2023-10-15'
        )
        
        assert isinstance(support_level, float)
        assert support_level == self.sample_data['Low'].min()


class TestChartDataPreparation:
    """Test chart data preparation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = PatternChartVisualizer(charts_dir=self.temp_dir)
    
    @patch('pattern_visualizer._load_cached_data')
    def test_prepare_match_chart_data_from_cache(self, mock_load_cached):
        """Test data preparation using cached data."""
        # Mock cached data
        dates = pd.date_range('2023-09-15', periods=60, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 60),
            'High': np.random.uniform(150, 250, 60),
            'Low': np.random.uniform(90, 150, 60),
            'Close': np.random.uniform(100, 200, 60),
            'Volume': np.random.uniform(1000000, 5000000, 60)
        }, index=dates)
        
        mock_load_cached.return_value = mock_data
        
        result = self.visualizer._prepare_match_chart_data(
            '0700.HK',
            '2023-10-01',
            '2023-10-31',
            buffer_days=10,
            context_days=5
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(col in result.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        mock_load_cached.assert_called_once_with('0700.HK')
    
    @patch('pattern_visualizer.fetch_hk_stocks')
    @patch('pattern_visualizer._load_cached_data')
    def test_prepare_match_chart_data_fetch_fallback(self, mock_load_cached, mock_fetch):
        """Test data preparation fallback to fetching new data."""
        mock_load_cached.return_value = None  # No cached data
        
        # Mock fetched data
        dates = pd.date_range('2023-09-15', periods=60, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 60),
            'High': np.random.uniform(150, 250, 60),
            'Low': np.random.uniform(90, 150, 60),
            'Close': np.random.uniform(100, 200, 60),
            'Volume': np.random.uniform(1000000, 5000000, 60)
        }, index=dates)
        
        mock_fetch.return_value = {'0700.HK': mock_data}
        
        result = self.visualizer._prepare_match_chart_data(
            '0700.HK',
            '2023-10-01',
            '2023-10-31'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        mock_fetch.assert_called_once()
    
    @patch('pattern_visualizer.fetch_hk_stocks')
    @patch('pattern_visualizer._load_cached_data')
    def test_prepare_match_chart_data_no_data(self, mock_load_cached, mock_fetch):
        """Test error handling when no data is available."""
        mock_load_cached.return_value = None
        mock_fetch.return_value = {}  # No data returned
        
        with pytest.raises(MatchVisualizationError, match="No data available"):
            self.visualizer._prepare_match_chart_data(
                '0700.HK',
                '2023-10-01',
                '2023-10-31'
            )


class TestVisualizationMethods:
    """Test the main visualization methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = PatternChartVisualizer(charts_dir=self.temp_dir)
        
        self.sample_match = {
            'ticker': '0700.HK',
            'window_start_date': '2023-10-01',
            'window_end_date': '2023-10-31',
            'confidence_score': 0.85,
            'rank': 1
        }
    
    @patch('pattern_visualizer.PatternChartVisualizer._prepare_match_chart_data')
    @patch('pattern_visualizer.PatternChartVisualizer._create_match_chart')
    @patch('pattern_visualizer.plt.show')
    def test_visualize_pattern_match_success(self, mock_show, mock_create_chart, mock_prepare_data):
        """Test successful single match visualization."""
        # Mock data preparation
        dates = pd.date_range('2023-09-20', periods=50, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 50),
            'High': np.random.uniform(150, 250, 50),
            'Low': np.random.uniform(90, 150, 50),
            'Close': np.random.uniform(100, 200, 50),
            'Volume': np.random.uniform(1000000, 5000000, 50)
        }, index=dates)
        
        mock_prepare_data.return_value = mock_data
        
        # Execute visualization
        self.visualizer.visualize_pattern_match(self.sample_match)
        
        # Verify method calls
        mock_prepare_data.assert_called_once()
        mock_create_chart.assert_called_once()
        mock_show.assert_called_once()
    
    def test_visualize_pattern_match_invalid_input(self):
        """Test error handling for invalid match input."""
        invalid_match = "invalid_input"
        
        with pytest.raises(MatchVisualizationError, match="Invalid match_row type"):
            self.visualizer.visualize_pattern_match(invalid_match)
    
    @patch('pattern_visualizer.PatternChartVisualizer.visualize_pattern_match')
    def test_visualize_all_matches(self, mock_visualize_single):
        """Test batch visualization of multiple matches."""
        matches_df = pd.DataFrame([
            {
                'ticker': '0700.HK',
                'window_start_date': '2023-10-01',
                'window_end_date': '2023-10-31',
                'confidence_score': 0.85,
                'rank': 1
            },
            {
                'ticker': '0005.HK',
                'window_start_date': '2023-09-15',
                'window_end_date': '2023-10-15',
                'confidence_score': 0.78,
                'rank': 2
            }
        ])
        
        self.visualizer.visualize_all_matches(
            matches_df,
            max_matches=2,
            min_confidence=0.7
        )
        
        # Should call visualization for each match
        assert mock_visualize_single.call_count == 2
    
    def test_visualize_all_matches_confidence_filter(self):
        """Test confidence filtering in batch visualization."""
        matches_df = pd.DataFrame([
            {
                'ticker': '0700.HK',
                'window_start_date': '2023-10-01',
                'window_end_date': '2023-10-31',
                'confidence_score': 0.85,
                'rank': 1
            },
            {
                'ticker': '0005.HK',
                'window_start_date': '2023-09-15',
                'window_end_date': '2023-10-15',
                'confidence_score': 0.60,  # Below threshold
                'rank': 2
            }
        ])
        
        with patch.object(self.visualizer, 'visualize_pattern_match') as mock_viz:
            self.visualizer.visualize_all_matches(
                matches_df,
                max_matches=10,
                min_confidence=0.7  # Should filter out second match
            )
            
            # Should only visualize the first match
            assert mock_viz.call_count == 1
    
    def test_generate_match_save_path(self):
        """Test chart save path generation."""
        save_path = self.visualizer._generate_match_save_path(
            '0700.HK',
            '2023-10-01',
            0.85
        )
        
        assert save_path.endswith('.png')
        assert '0700_HK' in save_path
        assert '20231001' in save_path
        assert 'conf085' in save_path
        assert save_path.startswith(self.temp_dir)


class TestConvenienceFunctions:
    """Test the convenience functions for notebook usage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_match = pd.Series({
            'ticker': '0700.HK',
            'window_start_date': '2023-10-01',
            'window_end_date': '2023-10-31',
            'confidence_score': 0.85,
            'rank': 1
        })
    
    @patch('pattern_visualizer.PatternChartVisualizer')
    def test_visualize_match_convenience_function(self, mock_visualizer_class):
        """Test the visualize_match convenience function."""
        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer
        
        visualize_match(self.sample_match, buffer_days=15)
        
        mock_visualizer_class.assert_called_once()
        mock_visualizer.visualize_pattern_match.assert_called_once_with(
            self.sample_match, buffer_days=15
        )
    
    @patch('pattern_visualizer.PatternChartVisualizer')
    def test_plot_match_convenience_function(self, mock_visualizer_class):
        """Test the plot_match convenience function."""
        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer
        
        plot_match(
            ticker='0700.HK',
            window_start='2023-10-01',
            window_end='2023-10-31',
            confidence_score=0.85,
            volume=True
        )
        
        mock_visualizer_class.assert_called_once()
        mock_visualizer.visualize_pattern_match.assert_called_once()
        
        # Check that match data was properly constructed
        call_args = mock_visualizer.visualize_pattern_match.call_args[0][0]
        assert call_args['ticker'] == '0700.HK'
        assert call_args['confidence_score'] == 0.85


class TestSummaryAndReporting:
    """Test summary report generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = PatternChartVisualizer(charts_dir=self.temp_dir)
        
        self.sample_matches = pd.DataFrame([
            {
                'ticker': '0700.HK',
                'window_start_date': '2023-10-01',
                'window_end_date': '2023-10-31',
                'confidence_score': 0.95,
                'rank': 1
            },
            {
                'ticker': '0005.HK',
                'window_start_date': '2023-09-15',
                'window_end_date': '2023-10-15',
                'confidence_score': 0.78,
                'rank': 2
            },
            {
                'ticker': '0700.HK',
                'window_start_date': '2023-11-01',
                'window_end_date': '2023-11-30',
                'confidence_score': 0.65,
                'rank': 3
            }
        ])
    
    def test_generate_match_summary_report(self):
        """Test comprehensive summary report generation."""
        summary = self.visualizer.generate_match_summary_report(
            self.sample_matches,
            save_report=False
        )
        
        assert summary['total_matches'] == 3
        assert summary['unique_tickers'] == 2
        assert 'confidence_stats' in summary
        assert 'confidence_distribution' in summary
        assert 'top_tickers' in summary
        
        # Check confidence statistics
        conf_stats = summary['confidence_stats']
        assert conf_stats['mean'] == pytest.approx(0.793, abs=0.01)
        assert conf_stats['max'] == 0.95
        assert conf_stats['min'] == 0.65
        
        # Check confidence distribution
        conf_dist = summary['confidence_distribution']
        assert conf_dist['high_confidence_0_9'] == 1  # Only one >= 0.9
        assert conf_dist['medium_confidence_0_7_0_9'] == 1  # One in 0.7-0.9 range
        assert conf_dist['low_confidence_below_0_7'] == 1  # One below 0.7
    
    def test_generate_match_summary_report_save_file(self):
        """Test saving summary report to file."""
        summary = self.visualizer.generate_match_summary_report(
            self.sample_matches,
            save_report=True
        )
        
        # Check that report file was created
        report_files = [f for f in os.listdir(self.temp_dir) if f.startswith('match_summary_')]
        assert len(report_files) == 1
        
        # Verify file content
        report_path = os.path.join(self.temp_dir, report_files[0])
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Pattern Match Summary Report' in content
            assert 'Total Matches: 3' in content
            assert 'Unique Tickers: 2' in content


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = PatternChartVisualizer(charts_dir=self.temp_dir)
    
    def test_match_visualization_error_inheritance(self):
        """Test that MatchVisualizationError inherits from VisualizationError."""
        from pattern_visualizer import VisualizationError
        
        assert issubclass(MatchVisualizationError, VisualizationError)
        
        error = MatchVisualizationError("Test error")
        assert str(error) == "Test error"
    
    @patch('pattern_visualizer.PatternChartVisualizer._prepare_match_chart_data')
    def test_visualization_handles_data_preparation_error(self, mock_prepare_data):
        """Test error handling in visualization when data preparation fails."""
        mock_prepare_data.side_effect = MatchVisualizationError("Data preparation failed")
        
        sample_match = {
            'ticker': '0700.HK',
            'window_start_date': '2023-10-01',
            'window_end_date': '2023-10-31',
            'confidence_score': 0.85
        }
        
        with pytest.raises(MatchVisualizationError, match="Failed to visualize match"):
            self.visualizer.visualize_pattern_match(sample_match)
    
    def test_empty_matches_handling(self):
        """Test handling of empty matches DataFrames."""
        empty_matches = pd.DataFrame()
        
        # Should not raise error, should handle gracefully
        self.visualizer.visualize_all_matches(empty_matches)
        
        summary = self.visualizer.generate_match_summary_report(empty_matches, save_report=False)
        assert summary == {}


class TestPerformanceRequirements:
    """Test performance requirements and timing validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = PatternChartVisualizer(charts_dir=self.temp_dir)
    
    @patch('pattern_visualizer.PatternChartVisualizer._prepare_match_chart_data')
    @patch('pattern_visualizer.PatternChartVisualizer._create_match_chart')
    @patch('pattern_visualizer.plt.show')
    @patch('pattern_visualizer.time.time')
    def test_visualization_performance_timing(self, mock_time, mock_show, mock_create_chart, mock_prepare_data):
        """Test that visualization timing is properly measured and reported."""
        # Mock time progression
        mock_time.side_effect = [0.0, 0.5]  # 0.5 second execution
        
        # Mock data preparation
        dates = pd.date_range('2023-09-20', periods=50, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 50),
            'High': np.random.uniform(150, 250, 50),
            'Low': np.random.uniform(90, 150, 50),
            'Close': np.random.uniform(100, 200, 50),
            'Volume': np.random.uniform(1000000, 5000000, 50)
        }, index=dates)
        mock_prepare_data.return_value = mock_data
        
        sample_match = {
            'ticker': '0700.HK',
            'window_start_date': '2023-10-01',
            'window_end_date': '2023-10-31',
            'confidence_score': 0.85
        }
        
        # Should not raise performance warning for <1 second
        self.visualizer.visualize_pattern_match(sample_match)
        
        # Verify timing was measured
        assert mock_time.call_count == 2


if __name__ == '__main__':
    pytest.main([__file__]) 