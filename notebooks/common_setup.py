"""
Shared Setup Utilities for Hong Kong Stock Analysis Notebooks

This module provides standardized setup functionality for all notebooks in the project,
including path configuration, imports, warning filtering, and common utilities.

Usage:
    from common_setup import setup_notebook, get_project_paths, configure_display
    setup_notebook()  # Sets up everything needed for most notebooks
"""

import sys
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

# Standard data science imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_project_paths() -> Dict[str, Path]:
    """
    Get standardized project paths for all notebooks.
    
    Returns:
        Dict with paths: project_root, src, data, models, features, etc.
    """
    notebook_dir = Path.cwd()
    
    # Handle both running from notebooks/ and project root
    if notebook_dir.name == 'notebooks':
        project_root = notebook_dir.parent
    else:
        project_root = notebook_dir
        
    paths = {
        'notebooks': project_root / 'notebooks',
        'project_root': project_root,
        'src': project_root / 'src',
        'data': project_root / 'data',
        'models': project_root / 'models',
        'features': project_root / 'features',
        'signals': project_root / 'signals',
        'charts': project_root / 'charts',
        'labels': project_root / 'labels',
        'temp_features': project_root / 'temp_features'
    }
    
    return paths

def configure_python_path() -> None:
    """Configure Python path to include src directory."""
    paths = get_project_paths()
    src_path = str(paths['src'])
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

def configure_warnings() -> None:
    """Configure warning filters to show important warnings while suppressing noise."""
    # Suppress specific noisy warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='yfinance')
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy')
    
    # Keep important warnings visible
    warnings.filterwarnings('default', category=RuntimeWarning)
    warnings.filterwarnings('default', category=ImportWarning)

def configure_logging(verbose: bool = False) -> None:
    """
    Configure logging levels to prevent excessive output during bulk operations.
    
    Args:
        verbose: If True, show detailed logs. If False, minimize output.
    """
    import logging
    
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    else:
        # Suppress most logging to keep output clean
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
        
        # Suppress specific noisy loggers
        logging.getLogger('yfinance').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)

def configure_display() -> None:
    """Configure pandas and matplotlib display options."""
    # Pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', 100)
    
    # Matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")

def create_directories() -> None:
    """Create necessary project directories if they don't exist."""
    paths = get_project_paths()
    
    directories_to_create = ['data', 'models', 'features', 'signals', 'charts', 'labels', 'temp_features']
    
    for dir_name in directories_to_create:
        dir_path = paths[dir_name]
        dir_path.mkdir(exist_ok=True)

def validate_environment() -> Dict[str, bool]:
    """
    Validate that the notebook environment is properly configured.
    
    Returns:
        Dict with validation results
    """
    paths = get_project_paths()
    
    validation = {
        'src_directory_exists': paths['src'].exists(),
        'src_in_python_path': str(paths['src']) in sys.path,
        'data_directory_exists': paths['data'].exists(),
        'required_imports_available': True
    }
    
    # Test critical imports
    try:
        from data_fetcher import fetch_hk_stocks
        from feature_extractor import FeatureExtractor
    except ImportError:
        validation['required_imports_available'] = False
    
    return validation

def setup_notebook(
    configure_paths: bool = True,
    configure_display_options: bool = True,
    create_dirs: bool = True,
    validate_env: bool = True,
    quiet: bool = False,
    verbose_logging: bool = False
) -> Optional[Dict]:
    """
    Complete notebook setup with all standard configurations.
    
    Args:
        configure_paths: Whether to configure Python paths
        configure_display_options: Whether to configure pandas/matplotlib
        create_dirs: Whether to create necessary directories
        validate_env: Whether to validate environment
        quiet: Whether to suppress setup messages
        verbose_logging: Whether to enable verbose logging for bulk operations
        
    Returns:
        Validation results if validate_env=True, None otherwise
    """
    if not quiet:
        print("🔧 Setting up notebook environment...")
    
    # Configure warnings and logging first
    configure_warnings()
    configure_logging(verbose=verbose_logging)
    
    # Configure Python path
    if configure_paths:
        configure_python_path()
        if not quiet:
            print("✅ Python path configured")
    
    # Configure display options
    if configure_display_options:
        configure_display()
        if not quiet:
            print("✅ Display options configured")
    
    # Create directories
    if create_dirs:
        create_directories()
        if not quiet:
            print("✅ Project directories ready")
    
    # Validate environment
    validation_results = None
    if validate_env:
        validation_results = validate_environment()
        if not quiet:
            all_valid = all(validation_results.values())
            status = "✅" if all_valid else "⚠️"
            print(f"{status} Environment validation {'passed' if all_valid else 'has issues'}")
            
            if not all_valid:
                for check, result in validation_results.items():
                    if not result:
                        print(f"   ❌ {check.replace('_', ' ').title()}")
    
    if not quiet:
        log_status = "verbose" if verbose_logging else "quiet"
        print(f"✅ Logging configured ({log_status} mode)")
        print(f"🎯 Notebook setup completed! ({datetime.now().strftime('%H:%M:%S')})")
    
    return validation_results

def get_date_range(days_back: int = 365) -> Tuple[str, str]:
    """
    Get standardized date range for data collection.
    
    Args:
        days_back: Number of days to go back from today
        
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    return start_date, end_date

def get_hk_stock_names() -> Dict[str, str]:
    """
    Get a dictionary of HK stock tickers and their company names.
    
    Returns:
        Dict mapping ticker symbols to company names
    """
    return {
        '0700.HK': 'Tencent Holdings Ltd',
        '0005.HK': 'HSBC Holdings plc',
        '0001.HK': 'CK Hutchison Holdings Ltd',
        '0388.HK': 'Hong Kong Exchanges and Clearing Ltd',
        '0003.HK': 'The Hong Kong and China Gas Company Ltd',
        '0939.HK': 'China Construction Bank Corporation',
        '1299.HK': 'AIA Group Ltd',
        '2318.HK': 'Ping An Insurance (Group) Company of China, Ltd.',
        '1398.HK': 'Industrial and Commercial Bank of China Ltd',
        '0883.HK': 'CNOOC Ltd',
        '6969.HK': 'Smoore International Holdings Ltd',
        '0016.HK': 'Sun Hung Kai Properties Ltd',
        '0175.HK': 'Geely Automobile Holdings Ltd',
        '0966.HK': 'China Taiping Insurance Holdings Company Ltd',
        '2020.HK': 'ANTA Sports Products Ltd'
    }

def print_setup_summary() -> None:
    """Print a summary of the current setup status."""
    paths = get_project_paths()
    validation = validate_environment()
    
    print("📋 **Notebook Environment Summary**")
    print("=" * 50)
    print(f"📁 Project Root: {paths['project_root']}")
    print(f"📁 Working Directory: {Path.cwd()}")
    print(f"🐍 Python Path: {str(paths['src']) in sys.path}")
    print(f"📊 Environment Valid: {all(validation.values())}")
    print(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show available data directories
    print(f"\n📂 **Available Directories:**")
    for name, path in paths.items():
        if path.exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} (missing)")

# Quick import helper for notebooks
def import_common_modules():
    """Import commonly used modules for notebooks."""
    try:
        from data_fetcher import fetch_hk_stocks, validate_tickers, get_all_cached_tickers
        from feature_extractor import FeatureExtractor
        from pattern_scanner import PatternScanner
        from hk_stock_universe import get_hk_stock_list_static
        
        return {
            'fetch_hk_stocks': fetch_hk_stocks,
            'validate_tickers': validate_tickers,
            'get_all_cached_tickers': get_all_cached_tickers,
            'FeatureExtractor': FeatureExtractor,
            'PatternScanner': PatternScanner,
            'get_hk_stock_list_static': get_hk_stock_list_static
        }
    except ImportError as e:
        print(f"⚠️ Could not import all common modules: {e}")
        return {}

def print_collection_summary(
    collected_data: Dict,
    failed_stocks: List[str],
    target_count: int,
    start_time: float,
    show_failed_details: bool = True,
    max_failed_shown: int = 10
) -> None:
    """
    Print a clean summary of bulk data collection results.
    
    Args:
        collected_data: Dictionary of successfully collected stock data
        failed_stocks: List of stock tickers that failed to collect
        target_count: Total number of stocks targeted
        start_time: Start time of collection (from time.time())
        show_failed_details: Whether to show individual failed stock names
        max_failed_shown: Maximum number of failed stocks to show individually
    """
    import time
    
    success_count = len(collected_data)
    failure_count = len(failed_stocks)
    success_rate = (success_count / target_count * 100) if target_count > 0 else 0
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"📊 **COLLECTION SUMMARY**")
    print(f"{'='*60}")
    print(f"✅ Successfully collected: {success_count:,} stocks")
    print(f"❌ Failed to collect: {failure_count:,} stocks")
    print(f"📈 Success rate: {success_rate:.1f}%")
    print(f"⏱️ Total time: {elapsed_time/60:.1f} minutes")
    
    if elapsed_time > 0:
        print(f"⚡ Rate: {success_count/(elapsed_time/60):.1f} stocks/minute")
    
    # Show failed stocks (with limit to avoid spam)
    if failure_count > 0 and show_failed_details:
        print(f"\n❌ **FAILED STOCKS:**")
        if failure_count <= max_failed_shown:
            print(f"   {', '.join(failed_stocks)}")
        else:
            shown = failed_stocks[:max_failed_shown]
            remaining = failure_count - max_failed_shown
            print(f"   {', '.join(shown)}")
            print(f"   ... and {remaining} more stocks")
    
    # Show data quality info
    if success_count > 0:
        # Calculate average data points per stock
        try:
            total_rows = sum(len(df) for df in collected_data.values() if df is not None and hasattr(df, '__len__'))
            avg_rows = total_rows / success_count if success_count > 0 else 0
            print(f"\n📈 **DATA QUALITY:**")
            print(f"   Total data points: {total_rows:,}")
            print(f"   Average per stock: {avg_rows:.0f} rows")
        except:
            # If we can't calculate data quality, skip it
            pass
    
    print(f"{'='*60}")

if __name__ == "__main__":
    # If run directly, perform setup and show summary
    setup_notebook(quiet=False)
    print_setup_summary() 