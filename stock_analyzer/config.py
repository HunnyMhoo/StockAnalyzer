"""
Central configuration module for stock_analyzer package.

This module provides centralized configuration management using Pydantic BaseSettings.
All configuration values can be overridden via environment variables with the 
STOCKANALYZER_ prefix.

Environment Variables:
    STOCKANALYZER_DATA_DIR: Data directory path
    STOCKANALYZER_ALERT_THRESHOLD: Pattern confidence alert threshold
    STOCKANALYZER_LINE_NOTIFY_TOKEN: LINE notification token
    STOCKANALYZER_SMA_WINDOWS: SMA window sizes (comma-separated)
    STOCKANALYZER_RSI_PERIOD: RSI calculation period
"""
import os
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration settings for stock_analyzer."""
    
    # Data storage configuration
    DATA_DIR: str = Field(default="data", description="Directory for storing stock data files")
    
    # Alert and notification configuration
    ALERT_THRESHOLD: float = Field(default=0.7, description="Minimum confidence threshold for pattern alerts")
    LINE_NOTIFY_TOKEN: str = Field(default="", description="LINE notification service token")
    
    # Technical analysis configuration
    SMA_WINDOWS: Union[List[int], str] = Field(default=[20, 50], description="Simple Moving Average window periods")
    RSI_PERIOD: int = Field(default=14, description="Relative Strength Index calculation period")
    
    # Pattern scanning configuration
    DEFAULT_WINDOW_SIZE: int = Field(default=30, description="Default window size for pattern scanning")
    DEFAULT_MAX_WINDOWS_PER_TICKER: int = Field(default=5, description="Default maximum windows per ticker")
    
    # Signal storage configuration
    SIGNALS_BASE_DIR: str = Field(default="signals", description="Base directory for signal storage")
    ENABLE_SIGNAL_COMPRESSION: bool = Field(default=True, description="Enable Parquet compression for signals")
    AUTO_SAVE_SIGNALS: bool = Field(default=True, description="Automatically save scanning results as signals")
    SIGNAL_RETENTION_DAYS: int = Field(default=365, description="Number of days to retain signal files")
    
    # Bollinger Bands configuration
    BOLLINGER_WINDOW: int = Field(default=20, description="Bollinger Bands calculation window")
    BOLLINGER_STD_DEV: float = Field(default=2.0, description="Bollinger Bands standard deviation multiplier")
    
    # Average True Range configuration
    ATR_PERIOD: int = Field(default=14, description="Average True Range calculation period")
    
    # Support/Resistance configuration
    SUPPORT_RESISTANCE_WINDOW: int = Field(default=5, description="Window for support/resistance detection")
    SUPPORT_RESISTANCE_MIN_PERIODS: int = Field(default=10, description="Minimum periods for support/resistance")
    
    # Volatility configuration
    VOLATILITY_WINDOW: int = Field(default=20, description="Price volatility calculation window")
    
    # Volume analysis configuration
    VOLUME_AVERAGE_WINDOW: int = Field(default=20, description="Volume average ratio calculation window")
    
    # Trend analysis configuration
    TREND_SLOPE_WINDOW: int = Field(default=30, description="Linear trend slope calculation window")
    
    @field_validator('SMA_WINDOWS')
    @classmethod
    def parse_sma_windows(cls, v):
        """Parse SMA_WINDOWS from comma-separated string if provided as env var."""
        if isinstance(v, str):
            try:
                # Try to parse as JSON first (for arrays like [10,20,30])
                import json
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [int(x) for x in parsed]
                else:
                    raise ValueError("JSON must be a list")
            except (json.JSONDecodeError, ValueError):
                # Fall back to comma-separated parsing
                result = []
                for x in v.split(','):
                    x = x.strip()
                    if x:  # Skip empty strings
                        try:
                            result.append(int(x))
                        except ValueError:
                            raise ValueError(f"Invalid integer value in SMA_WINDOWS: '{x}'")
                return result
        elif isinstance(v, list):
            return [int(x) for x in v]
        return v
    
    @field_validator('ALERT_THRESHOLD')
    @classmethod
    def validate_alert_threshold(cls, v):
        """Ensure alert threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError('ALERT_THRESHOLD must be between 0 and 1')
        return v
    
    @field_validator('RSI_PERIOD', 'DEFAULT_WINDOW_SIZE', 'DEFAULT_MAX_WINDOWS_PER_TICKER', 
                     'BOLLINGER_WINDOW', 'ATR_PERIOD', 'SUPPORT_RESISTANCE_WINDOW',
                     'SUPPORT_RESISTANCE_MIN_PERIODS', 'VOLATILITY_WINDOW', 
                     'VOLUME_AVERAGE_WINDOW', 'TREND_SLOPE_WINDOW')
    @classmethod
    def validate_positive_integers(cls, v):
        """Ensure all period/window values are positive integers."""
        if v <= 0:
            raise ValueError('Period and window values must be positive integers')
        return v
    
    @field_validator('SMA_WINDOWS')
    @classmethod
    def validate_sma_windows(cls, v):
        """Ensure all SMA windows are positive integers."""
        if not all(isinstance(w, int) and w > 0 for w in v):
            raise ValueError('All SMA windows must be positive integers')
        return v

    class Config:
        """Pydantic configuration."""
        env_prefix = 'STOCKANALYZER_'
        case_sensitive = True
        env_file = '.env'
        env_file_encoding = 'utf-8'


# Singleton settings instance
settings = Settings() 