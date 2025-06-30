"""
Test cases for environment variable override functionality in configuration module.

This module tests that environment variables can properly override default 
configuration values in the stock_analyzer.config module.
"""
import os
import pytest
from unittest.mock import patch

from stock_analyzer.config import Settings


class TestEnvironmentOverride:
    """Test environment variable override functionality."""
    
    def test_data_dir_env_override(self, monkeypatch):
        """Test that STOCKANALYZER_DATA_DIR environment variable overrides default."""
        test_data_dir = "/tmp/sa_data"
        monkeypatch.setenv("STOCKANALYZER_DATA_DIR", test_data_dir)
        
        # Create new settings instance to pick up env var
        settings = Settings()
        
        assert settings.DATA_DIR == test_data_dir
        assert settings.DATA_DIR != "data"  # Should not be default
    
    def test_alert_threshold_env_override(self, monkeypatch):
        """Test that STOCKANALYZER_ALERT_THRESHOLD environment variable overrides default."""
        test_threshold = "0.85"
        monkeypatch.setenv("STOCKANALYZER_ALERT_THRESHOLD", test_threshold)
        
        settings = Settings()
        
        assert settings.ALERT_THRESHOLD == 0.85
        assert settings.ALERT_THRESHOLD != 0.7  # Should not be default
    
    def test_line_notify_token_env_override(self, monkeypatch):
        """Test that STOCKANALYZER_LINE_NOTIFY_TOKEN environment variable overrides default."""
        test_token = "test_token_12345"
        monkeypatch.setenv("STOCKANALYZER_LINE_NOTIFY_TOKEN", test_token)
        
        settings = Settings()
        
        assert settings.LINE_NOTIFY_TOKEN == test_token
        assert settings.LINE_NOTIFY_TOKEN != ""  # Should not be default
    
    def test_sma_windows_env_override(self, monkeypatch):
        """Test that STOCKANALYZER_SMA_WINDOWS environment variable overrides default."""
        test_windows = "10,25,50"
        monkeypatch.setenv("STOCKANALYZER_SMA_WINDOWS", test_windows)
        
        settings = Settings()
        
        assert settings.SMA_WINDOWS == [10, 25, 50]
        assert settings.SMA_WINDOWS != [20, 50]  # Should not be default
    
    def test_rsi_period_env_override(self, monkeypatch):
        """Test that STOCKANALYZER_RSI_PERIOD environment variable overrides default."""
        test_period = "21"
        monkeypatch.setenv("STOCKANALYZER_RSI_PERIOD", test_period)
        
        settings = Settings()
        
        assert settings.RSI_PERIOD == 21
        assert settings.RSI_PERIOD != 14  # Should not be default
    
    def test_default_window_size_env_override(self, monkeypatch):
        """Test that STOCKANALYZER_DEFAULT_WINDOW_SIZE environment variable overrides default."""
        test_window = "45"
        monkeypatch.setenv("STOCKANALYZER_DEFAULT_WINDOW_SIZE", test_window)
        
        settings = Settings()
        
        assert settings.DEFAULT_WINDOW_SIZE == 45
        assert settings.DEFAULT_WINDOW_SIZE != 30  # Should not be default
    
    def test_multiple_env_overrides(self, monkeypatch):
        """Test that multiple environment variables can be set simultaneously."""
        monkeypatch.setenv("STOCKANALYZER_DATA_DIR", "/custom/data")
        monkeypatch.setenv("STOCKANALYZER_ALERT_THRESHOLD", "0.8")
        monkeypatch.setenv("STOCKANALYZER_RSI_PERIOD", "10")
        monkeypatch.setenv("STOCKANALYZER_SMA_WINDOWS", "5,15,30")
        
        settings = Settings()
        
        assert settings.DATA_DIR == "/custom/data"
        assert settings.ALERT_THRESHOLD == 0.8
        assert settings.RSI_PERIOD == 10
        assert settings.SMA_WINDOWS == [5, 15, 30]
    
    def test_invalid_alert_threshold_validation(self, monkeypatch):
        """Test that invalid alert threshold values raise validation errors."""
        monkeypatch.setenv("STOCKANALYZER_ALERT_THRESHOLD", "1.5")
        
        with pytest.raises(ValueError, match="ALERT_THRESHOLD must be between 0 and 1"):
            Settings()
    
    def test_invalid_rsi_period_validation(self, monkeypatch):
        """Test that invalid RSI period values raise validation errors."""
        monkeypatch.setenv("STOCKANALYZER_RSI_PERIOD", "-5")
        
        with pytest.raises(ValueError, match="Period and window values must be positive integers"):
            Settings()
    
    def test_invalid_sma_windows_validation(self, monkeypatch):
        """Test that invalid SMA windows raise validation errors."""
        monkeypatch.setenv("STOCKANALYZER_SMA_WINDOWS", "10,-5,20")
        
        with pytest.raises(ValueError, match="All SMA windows must be positive integers"):
            Settings()
    
    def test_sma_windows_string_parsing(self, monkeypatch):
        """Test that SMA windows string is properly parsed."""
        # Test with spaces
        monkeypatch.setenv("STOCKANALYZER_SMA_WINDOWS", "10, 20, 30")
        settings = Settings()
        assert settings.SMA_WINDOWS == [10, 20, 30]
        
        # Test with no spaces
        monkeypatch.setenv("STOCKANALYZER_SMA_WINDOWS", "5,15,25,35")
        settings = Settings()
        assert settings.SMA_WINDOWS == [5, 15, 25, 35]
    
    def test_env_file_support(self, tmp_path, monkeypatch):
        """Test that .env file is properly loaded."""
        env_file = tmp_path / ".env"
        env_file.write_text("STOCKANALYZER_DATA_DIR=/env/file/data\n")
        
        # Change to temp directory to pick up .env file
        monkeypatch.chdir(tmp_path)
        
        settings = Settings()
        assert settings.DATA_DIR == "/env/file/data"


class TestDefaultValues:
    """Test that default values are properly set when no environment variables are provided."""
    
    def test_default_values_without_env(self):
        """Test that default values are used when no environment variables are set."""
        # Create settings without any environment variables
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            assert settings.DATA_DIR == "data"
            assert settings.ALERT_THRESHOLD == 0.7
            assert settings.LINE_NOTIFY_TOKEN == ""
            assert settings.SMA_WINDOWS == [20, 50]
            assert settings.RSI_PERIOD == 14
            assert settings.DEFAULT_WINDOW_SIZE == 30
            assert settings.DEFAULT_MAX_WINDOWS_PER_TICKER == 5
            assert settings.BOLLINGER_WINDOW == 20
            assert settings.BOLLINGER_STD_DEV == 2.0
            assert settings.ATR_PERIOD == 14
            assert settings.SUPPORT_RESISTANCE_WINDOW == 5
            assert settings.SUPPORT_RESISTANCE_MIN_PERIODS == 10
            assert settings.VOLATILITY_WINDOW == 20
            assert settings.VOLUME_AVERAGE_WINDOW == 20
            assert settings.TREND_SLOPE_WINDOW == 30 