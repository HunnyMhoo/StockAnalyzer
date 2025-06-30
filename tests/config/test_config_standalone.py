#!/usr/bin/env python3
"""
Standalone test for configuration module that avoids import issues.

This test directly executes the config module and tests its functionality
without importing the full stock_analyzer package.
"""
import os
import sys
import tempfile
from pathlib import Path


def test_config_standalone():
    """Test configuration module functionality directly."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / "stock_analyzer" / "config.py"
    
    print(f"Testing config file: {config_file}")
    assert config_file.exists(), f"Config file not found: {config_file}"
    
    # Test 1: Default values
    print("\n=== Test 1: Default Values ===")
    
    # Clear any existing env vars
    env_vars_to_clear = [k for k in os.environ.keys() if k.startswith('STOCKANALYZER_')]
    for var in env_vars_to_clear:
        del os.environ[var]
    
    # Execute the config module
    config_globals = {}
    exec(config_file.read_text(), config_globals)
    settings = config_globals['settings']
    
    # Test default values
    assert settings.DATA_DIR == "data", f"Expected 'data', got '{settings.DATA_DIR}'"
    assert settings.ALERT_THRESHOLD == 0.7, f"Expected 0.7, got {settings.ALERT_THRESHOLD}"
    assert settings.LINE_NOTIFY_TOKEN == "", f"Expected '', got '{settings.LINE_NOTIFY_TOKEN}'"
    assert settings.SMA_WINDOWS == [20, 50], f"Expected [20, 50], got {settings.SMA_WINDOWS}"
    assert settings.RSI_PERIOD == 14, f"Expected 14, got {settings.RSI_PERIOD}"
    
    print("✓ All default values correct")
    
    # Test 2: Environment variable overrides
    print("\n=== Test 2: Environment Variable Overrides ===")
    
    os.environ['STOCKANALYZER_DATA_DIR'] = '/tmp/sa_data'
    os.environ['STOCKANALYZER_ALERT_THRESHOLD'] = '0.85'
    os.environ['STOCKANALYZER_LINE_NOTIFY_TOKEN'] = 'test_token_123'
    os.environ['STOCKANALYZER_SMA_WINDOWS'] = '10,25,50'
    os.environ['STOCKANALYZER_RSI_PERIOD'] = '21'
    
    # Execute the config module with env vars
    config_globals = {}
    exec(config_file.read_text(), config_globals)
    settings = config_globals['settings']
    
    # Test overridden values
    assert settings.DATA_DIR == '/tmp/sa_data', f"Expected '/tmp/sa_data', got '{settings.DATA_DIR}'"
    assert settings.ALERT_THRESHOLD == 0.85, f"Expected 0.85, got {settings.ALERT_THRESHOLD}"
    assert settings.LINE_NOTIFY_TOKEN == 'test_token_123', f"Expected 'test_token_123', got '{settings.LINE_NOTIFY_TOKEN}'"
    assert settings.SMA_WINDOWS == [10, 25, 50], f"Expected [10, 25, 50], got {settings.SMA_WINDOWS}"
    assert settings.RSI_PERIOD == 21, f"Expected 21, got {settings.RSI_PERIOD}"
    
    print("✓ All environment variable overrides work correctly")
    
    # Test 3: Validation errors
    print("\n=== Test 3: Validation Errors ===")
    
    # Test invalid alert threshold
    os.environ['STOCKANALYZER_ALERT_THRESHOLD'] = '1.5'
    
    try:
        config_globals = {}
        exec(config_file.read_text(), config_globals)
        assert False, "Expected validation error for ALERT_THRESHOLD > 1"
    except ValueError as e:
        assert "ALERT_THRESHOLD must be between 0 and 1" in str(e)
        print("✓ Alert threshold validation works")
    
    # Test invalid RSI period
    os.environ['STOCKANALYZER_ALERT_THRESHOLD'] = '0.8'  # Reset to valid value
    os.environ['STOCKANALYZER_RSI_PERIOD'] = '-5'
    
    try:
        config_globals = {}
        exec(config_file.read_text(), config_globals)
        assert False, "Expected validation error for negative RSI_PERIOD"
    except ValueError as e:
        assert "Period and window values must be positive integers" in str(e)
        print("✓ RSI period validation works")
    
    # Test invalid SMA windows
    os.environ['STOCKANALYZER_RSI_PERIOD'] = '14'  # Reset to valid value
    os.environ['STOCKANALYZER_SMA_WINDOWS'] = '10,-5,20'
    
    try:
        config_globals = {}
        exec(config_file.read_text(), config_globals)
        assert False, "Expected validation error for negative SMA window"
    except ValueError as e:
        assert "All SMA windows must be positive integers" in str(e)
        print("✓ SMA windows validation works")
    
    # Clean up environment variables
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
    for var in ['STOCKANALYZER_DATA_DIR', 'STOCKANALYZER_ALERT_THRESHOLD', 
                'STOCKANALYZER_LINE_NOTIFY_TOKEN', 'STOCKANALYZER_SMA_WINDOWS', 
                'STOCKANALYZER_RSI_PERIOD']:
        if var in os.environ:
            del os.environ[var]
    
    print("\n=== All Tests Passed! ===")
    print("✓ Configuration module working correctly")
    print("✓ Default values are correct")  
    print("✓ Environment variable overrides work")
    print("✓ Validation errors are caught properly")


if __name__ == "__main__":
    test_config_standalone() 