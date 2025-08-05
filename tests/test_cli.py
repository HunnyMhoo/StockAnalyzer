"""
Smoke tests for the CLI functionality.

These tests validate that the CLI commands are properly installed,
argument parsing works correctly, and help output is displayed.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_installed():
    """Test that the CLI is properly installed and discoverable."""
    result = subprocess.run(
        ["stock-analyzer", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    assert result.returncode == 0
    assert "Hong Kong Stock Pattern Recognition System CLI" in result.stdout
    assert "fetch" in result.stdout
    assert "features" in result.stdout
    assert "scan" in result.stdout
    assert "train" in result.stdout


def test_fetch_command_help():
    """Test that the fetch command shows proper help output."""
    result = subprocess.run(
        ["stock-analyzer", "fetch", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    assert result.returncode == 0
    assert "Fetch bulk stock data for multiple tickers" in result.stdout
    assert "--start-date" in result.stdout
    assert "--end-date" in result.stdout
    assert "--batch-size" in result.stdout
    assert "TICKERS..." in result.stdout


def test_features_command_help():
    """Test that the features command shows proper help output."""
    result = subprocess.run(
        ["stock-analyzer", "features", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    assert result.returncode == 0
    assert "Extract technical features from labeled pattern data" in result.stdout
    assert "--window-size" in result.stdout
    assert "--output-dir" in result.stdout
    assert "LABELED_DATA_PATH" in result.stdout


def test_scan_command_help():
    """Test that the scan command shows proper help output."""
    result = subprocess.run(
        ["stock-analyzer", "scan", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    assert result.returncode == 0
    assert "Scan multiple stocks for trading patterns" in result.stdout
    assert "--model" in result.stdout
    assert "--min-confidence" in result.stdout
    assert "TICKERS..." in result.stdout


def test_train_command_help():
    """Test that the train command shows proper help output."""
    result = subprocess.run(
        ["stock-analyzer", "train", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    assert result.returncode == 0
    assert "Train a pattern detection model" in result.stdout
    assert "--model-type" in result.stdout
    assert "--test-size" in result.stdout
    assert "LABELED_DATA_PATH" in result.stdout


def test_version_command():
    """Test that the version command shows the correct version."""
    result = subprocess.run(
        ["stock-analyzer", "version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    assert result.returncode == 0
    assert "stock-analyzer version 0.1.0" in result.stdout


def test_fetch_command_missing_arguments():
    """Test that fetch command fails gracefully with missing arguments."""
    result = subprocess.run(
        ["stock-analyzer", "fetch"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    assert result.returncode != 0
    assert "Missing argument" in result.stderr or "Error" in result.stderr


def test_features_command_missing_arguments():
    """Test that features command fails gracefully with missing arguments."""
    result = subprocess.run(
        ["stock-analyzer", "features"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    assert result.returncode != 0
    assert "Missing argument" in result.stderr or "Error" in result.stderr


def test_scan_command_missing_arguments():
    """Test that scan command fails gracefully with missing arguments."""
    result = subprocess.run(
        ["stock-analyzer", "scan"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    assert result.returncode != 0
    assert "Missing argument" in result.stderr or "Error" in result.stderr


def test_train_command_missing_arguments():
    """Test that train command fails gracefully with missing arguments."""
    result = subprocess.run(
        ["stock-analyzer", "train"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    assert result.returncode != 0
    assert "Missing argument" in result.stderr or "Error" in result.stderr


def test_cli_import():
    """Test that the CLI module can be imported and has the expected structure."""
    from stock_analyzer.cli import app
    
    assert app is not None
    assert hasattr(app, "registered_commands")
    
    # Check that commands are registered (the actual command names are tested via subprocess)
    assert len(app.registered_commands) >= 5, f"Expected at least 5 commands, got {len(app.registered_commands)}"


def test_cli_delegates_to_existing_functions():
    """Test that CLI commands delegate to existing functions (no new logic)."""
    from stock_analyzer.cli import fetch, features, scan, train
    
    # Import the actual functions to verify they exist
    from stock_analyzer import fetch_bulk, build_features, scan_patterns, train_model
    
    # Verify that the CLI functions are different from the public API functions
    # (they should be wrapper functions that call the public API)
    assert fetch != fetch_bulk
    assert features != build_features
    assert scan != scan_patterns
    assert train != train_model
    
    # Verify that the CLI functions are callable
    assert callable(fetch)
    assert callable(features)
    assert callable(scan)
    assert callable(train) 