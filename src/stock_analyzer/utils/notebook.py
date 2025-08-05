"""
Notebook utilities for reproducible, tidy notebooks with standard UX.

This module provides bootstrap functionality to ensure consistent notebook setup,
including path configuration, display settings, and lightweight logging.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd


class NotebookLogger:
    """
    Lightweight logger for notebook operations.
    
    Provides simple logging functionality without heavy dependencies,
    suitable for notebook environments.
    """
    
    def __init__(self, name: str = "notebook", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self._setup_complete = False
    
    def info(self, message: str) -> None:
        """Log an info message."""
        if self.verbose:
            print(f"ℹ️  [{self.name}] {message}")
    
    def success(self, message: str) -> None:
        """Log a success message."""
        if self.verbose:
            print(f"✅ [{self.name}] {message}")
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        if self.verbose:
            print(f"⚠️  [{self.name}] {message}")
    
    def error(self, message: str) -> None:
        """Log an error message."""
        print(f"❌ [{self.name}] {message}")
    
    def setup_complete(self) -> None:
        """Mark setup as complete."""
        self._setup_complete = True
        self.success("Notebook setup complete")
    
    @property
    def is_setup_complete(self) -> bool:
        """Check if setup is complete."""
        return self._setup_complete


def _add_src_to_path() -> None:
    """Add src directory to Python path."""
    current_dir = Path.cwd()
    
    # Find project root by looking for key indicators
    project_root = current_dir
    for parent in [current_dir] + list(current_dir.parents):
        if any(
            indicator.exists()
            for indicator in [
                parent / "pyproject.toml",
                parent / "src",
                parent / "stock_analyzer",
            ]
        ):
            project_root = parent
            break
    
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _configure_pandas_display() -> None:
    """Configure pandas display options for better notebook UX."""
    # Set display options for better readability
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.precision", 4)
    
    # Enable float formatting for better readability
    pd.set_option("display.float_format", lambda x: "%.4f" % x if abs(x) < 1e-4 or abs(x) >= 1e4 else "%.6f" % x)


def _configure_warnings() -> None:
    """Configure warning filters for cleaner notebook output."""
    # Suppress common noisy warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")


def _enable_autoreload() -> None:
    """Enable IPython autoreload for development."""
    try:
        from IPython.core.getipython import get_ipython
        
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("load_ext", "autoreload")
            ipython.run_line_magic("autoreload", "2")
    except ImportError:
        # Not in IPython environment, skip autoreload
        pass


def _validate_environment() -> Dict[str, bool]:
    """Validate the notebook environment."""
    checks = {
        "pandas_available": False,
        "numpy_available": False,
        "matplotlib_available": False,
        "stock_analyzer_available": False,
    }
    
    try:
        import pandas as pd
        checks["pandas_available"] = True
    except ImportError:
        pass
    
    try:
        import numpy as np
        checks["numpy_available"] = True
    except ImportError:
        pass
    
    try:
        import matplotlib.pyplot as plt
        checks["matplotlib_available"] = True
    except ImportError:
        pass
    
    try:
        import stock_analyzer
        checks["stock_analyzer_available"] = True
    except ImportError:
        pass
    
    return checks


def bootstrap(
    verbose: bool = True,
    enable_autoreload: bool = True,
    configure_display: bool = True,
    configure_warnings: bool = True,
) -> Dict[str, Any]:
    """
    Bootstrap function for reproducible, tidy notebooks.
    
    This function sets up the notebook environment with:
    - Python path configuration (adds src to path)
    - Pandas display options for better UX
    - Warning filters for cleaner output
    - Autoreload for development (optional)
    - Environment validation
    
    Args:
        verbose: Whether to show setup messages
        enable_autoreload: Whether to enable IPython autoreload
        configure_display: Whether to configure pandas display options
        configure_warnings: Whether to configure warning filters
        
    Returns:
        Dict containing setup information and validation results
    """
    logger = NotebookLogger("bootstrap", verbose=verbose)
    
    logger.info("Starting notebook bootstrap...")
    
    # Add src to Python path
    logger.info("Configuring Python path...")
    _add_src_to_path()
    
    # Configure pandas display
    if configure_display:
        logger.info("Configuring pandas display options...")
        _configure_pandas_display()
    
    # Configure warnings
    if configure_warnings:
        logger.info("Configuring warning filters...")
        _configure_warnings()
    
    # Enable autoreload
    if enable_autoreload:
        logger.info("Enabling autoreload...")
        _enable_autoreload()
    
    # Validate environment
    logger.info("Validating environment...")
    validation = _validate_environment()
    
    # Log validation results
    for check, result in validation.items():
        if result:
            logger.success(f"{check}: ✅")
        else:
            logger.warning(f"{check}: ❌")
    
    # Create result dict
    result = {
        "validation": validation,
        "paths": {
            "current_dir": str(Path.cwd()),
            "python_path": sys.path[:3],  # First 3 entries
        },
        "settings": {
            "verbose": verbose,
            "autoreload_enabled": enable_autoreload,
            "display_configured": configure_display,
            "warnings_configured": configure_warnings,
        }
    }
    
    logger.setup_complete()
    
    return result


def quick_bootstrap() -> None:
    """
    Quick bootstrap with default settings.
    
    Convenience function for one-line setup in notebooks.
    """
    bootstrap(verbose=True, enable_autoreload=True)


def minimal_bootstrap() -> None:
    """
    Minimal bootstrap with essential settings only.
    
    For notebooks that need minimal setup.
    """
    bootstrap(verbose=False, enable_autoreload=False, configure_display=True, configure_warnings=True)


def get_project_paths() -> Dict[str, Path]:
    """
    Get standardized project paths.
    
    Returns:
        Dict with common project paths
    """
    current_dir = Path.cwd()
    
    # Find project root
    project_root = current_dir
    for parent in [current_dir] + list(current_dir.parents):
        if any(
            indicator.exists()
            for indicator in [
                parent / "pyproject.toml",
                parent / "src",
                parent / "stock_analyzer",
            ]
        ):
            project_root = parent
            break
    
    return {
        "project_root": project_root,
        "src": project_root / "src",
        "data": project_root / "data",
        "models": project_root / "models",
        "features": project_root / "features",
        "signals": project_root / "signals",
        "charts": project_root / "charts",
        "labels": project_root / "labels",
        "temp_features": project_root / "temp_features",
        "notebooks": project_root / "notebooks",
    }


def create_directories() -> None:
    """Create common project directories if they don't exist."""
    paths = get_project_paths()
    
    for name, path in paths.items():
        if name != "project_root" and name != "src":
            path.mkdir(parents=True, exist_ok=True)


def print_setup_summary() -> None:
    """Print a summary of the current setup."""
    paths = get_project_paths()
    
    print("📁 Project Paths:")
    for name, path in paths.items():
        status = "✅" if path.exists() else "❌"
        print(f"   {name}: {status} {path}")
    
    print(f"\n🐍 Python Path (first 3 entries):")
    for i, path in enumerate(sys.path[:3]):
        print(f"   {i}: {path}")


# Convenience functions for common imports
def import_common_modules():
    """Import commonly used modules."""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    return pd, np, plt, sns


def import_stock_analyzer_modules():
    """Import stock analyzer modules."""
    from stock_analyzer import fetch_bulk, build_features, scan_patterns, train_model
    
    return fetch_bulk, build_features, scan_patterns, train_model
