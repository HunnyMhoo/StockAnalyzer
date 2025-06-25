"""
Jupyter Notebook Utilities

This module provides utilities to fix common Jupyter notebook issues like
output buffering, progress display, and immediate feedback.
"""

import sys
import time
from typing import Any, List, Dict
from IPython.display import display, clear_output, HTML
import pandas as pd


class NotebookLogger:
    """Logger that flushes output immediately in Jupyter notebooks."""
    
    def __init__(self, flush_immediately: bool = True):
        self.flush_immediately = flush_immediately
    
    def print(self, *args, **kwargs):
        """Print with immediate flush."""
        print(*args, **kwargs)
        if self.flush_immediately:
            sys.stdout.flush()
    
    def print_status(self, message: str):
        """Print status message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        self.print(f"[{timestamp}] {message}")
    
    def print_progress(self, current: int, total: int, message: str = ""):
        """Print progress with percentage."""
        percentage = (current / total) * 100 if total > 0 else 0
        progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
        status = f"[{progress_bar}] {percentage:.1f}% ({current}/{total})"
        if message:
            status += f" - {message}"
        self.print(status)


def immediate_feedback_wrapper(func):
    """Decorator to provide immediate feedback for long-running functions."""
    def wrapper(*args, **kwargs):
        logger = NotebookLogger()
        logger.print_status(f"🚀 Starting {func.__name__}...")
        
        try:
            result = func(*args, **kwargs)
            logger.print_status(f"✅ {func.__name__} completed successfully!")
            return result
        except Exception as e:
            logger.print_status(f"❌ {func.__name__} failed: {e}")
            raise
    
    return wrapper


def create_interactive_progress_bar(total: int, description: str = "Processing"):
    """Create an interactive progress bar for notebooks."""
    from tqdm.notebook import tqdm
    
    # Force immediate display
    sys.stdout.flush()
    
    return tqdm(
        total=total,
        desc=description,
        unit="items",
        ncols=100,
        leave=True,
        dynamic_ncols=True
    )


def quick_test_notebook_output():
    """Quick test to verify notebook output is working immediately."""
    logger = NotebookLogger()
    
    logger.print("🧪 Testing immediate output...")
    sys.stdout.flush()
    
    for i in range(3):
        logger.print(f"   Step {i+1}/3...")
        time.sleep(0.5)  # Short delay to see immediate output
    
    logger.print("✅ Notebook output test completed!")
    return True


def display_dataframe_summary(df: pd.DataFrame, title: str = "Data Summary"):
    """Display DataFrame with better formatting in notebooks."""
    logger = NotebookLogger()
    
    logger.print(f"\n📊 {title}")
    logger.print("=" * 50)
    logger.print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    if len(df) > 0:
        logger.print(f"Columns: {list(df.columns)}")
        display(df)
    else:
        logger.print("❌ No data to display")


def create_html_progress(current: int, total: int, message: str = ""):
    """Create HTML progress bar for notebooks."""
    percentage = (current / total) * 100 if total > 0 else 0
    
    html = f"""
    <div style="border: 1px solid #ccc; border-radius: 5px; padding: 10px; margin: 5px 0;">
        <div style="background-color: #4CAF50; height: 20px; width: {percentage}%; border-radius: 3px;"></div>
        <p style="margin: 5px 0 0 0; font-family: monospace;">
            {percentage:.1f}% ({current}/{total}) {message}
        </p>
    </div>
    """
    
    display(HTML(html))


def notebook_friendly_fetch_demo():
    """Demo function that shows immediate output in notebooks."""
    logger = NotebookLogger()
    
    logger.print("🚀 Starting notebook-friendly demo...")
    logger.print("💡 You should see each step immediately!")
    
    steps = [
        "Initializing connection...",
        "Validating parameters...", 
        "Fetching sample data...",
        "Processing results...",
        "Generating summary..."
    ]
    
    for i, step in enumerate(steps):
        logger.print_progress(i + 1, len(steps), step)
        time.sleep(1)  # Simulate work
    
    logger.print("✅ Demo completed successfully!")
    
    # Return some sample data
    sample_data = pd.DataFrame({
        'Ticker': ['0700.HK', '0005.HK', '0941.HK'],
        'Status': ['Success', 'Success', 'Success'],
        'Records': [252, 252, 252]
    })
    
    display_dataframe_summary(sample_data, "Demo Results")
    return sample_data


# Make logger available globally for imports
nb_logger = NotebookLogger()


def nb_print(*args, **kwargs):
    """Notebook-friendly print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


if __name__ == "__main__":
    # Run tests
    print("🧪 Testing notebook utilities...")
    quick_test_notebook_output()
    print("\n" + "="*50)
    notebook_friendly_fetch_demo() 