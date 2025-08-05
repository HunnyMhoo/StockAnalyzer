"""
Command-line interface for the Stock Analyzer package.

This module provides CLI commands for the main stock analyzer functions,
allowing headless execution of the same workflows available in notebooks.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import fetch_bulk, build_features, scan_patterns, train_model
from .utils.errors import DataFetchError, FeatureExtractionError, PatternScanningError, ModelTrainingError

# Create Typer app
app = typer.Typer(
    name="stock-analyzer",
    help="Hong Kong Stock Pattern Recognition System CLI",
    add_completion=False,
)

# Rich console for better output
console = Console()


def _print_success(message: str) -> None:
    """Print a success message with green color."""
    console.print(f"✅ {message}", style="green")


def _print_error(message: str) -> None:
    """Print an error message with red color."""
    console.print(f"❌ {message}", style="red")


def _print_info(message: str) -> None:
    """Print an info message with blue color."""
    console.print(f"ℹ️  {message}", style="blue")


@app.command()
def fetch(
    tickers: List[str] = typer.Argument(..., help="List of stock ticker symbols (e.g., 0700.HK 0941.HK)"),
    start_date: str = typer.Option(..., "--start-date", "-s", help="Start date in YYYY-MM-DD format"),
    end_date: str = typer.Option(..., "--end-date", "-e", help="End date in YYYY-MM-DD format"),
    batch_size: int = typer.Option(20, "--batch-size", "-b", help="Number of stocks to process per batch"),
    delay_between_batches: float = typer.Option(2.0, "--delay", "-d", help="Delay in seconds between batches"),
    max_retries: int = typer.Option(2, "--retries", "-r", help="Number of retries for failed stocks"),
    force_refresh: bool = typer.Option(False, "--force-refresh", "-f", help="Ignore cache and fetch fresh data"),
    skip_failed: bool = typer.Option(True, "--skip-failed", help="Continue processing even if some stocks fail"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file to save results (JSON)"),
) -> None:
    """
    Fetch bulk stock data for multiple tickers.
    
    This command fetches historical stock data for the specified tickers
    and saves the results to a file or displays summary information.
    """
    try:
        _print_info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching stock data...", total=None)
            
            results = fetch_bulk(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                batch_size=batch_size,
                delay_between_batches=delay_between_batches,
                max_retries=max_retries,
                force_refresh=force_refresh,
                skip_failed=skip_failed,
            )
            
            progress.update(task, completed=True)
        
        # Display results summary
        table = Table(title="Fetch Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Tickers", str(len(tickers)))
        table.add_row("Successful Fetches", str(results.get("successful_count", 0)))
        table.add_row("Failed Fetches", str(results.get("failed_count", 0)))
        table.add_row("Total Data Points", str(results.get("total_data_points", 0)))
        
        console.print(table)
        
        # Save results if output file specified
        if output_file:
            output_path = Path(output_file)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            _print_success(f"Results saved to {output_path}")
        
        _print_success("Data fetching completed successfully!")
        
    except DataFetchError as e:
        _print_error(f"Data fetching failed: {e}")
        sys.exit(1)
    except Exception as e:
        _print_error(f"Unexpected error: {e}")
        sys.exit(1)


@app.command()
def features(
    labeled_data_path: str = typer.Argument(..., help="Path to the labeled pattern data file"),
    window_size: int = typer.Option(30, "--window-size", "-w", help="Size of the feature extraction window"),
    prior_context_days: int = typer.Option(30, "--prior-context", "-p", help="Number of days for prior context"),
    support_lookback_days: int = typer.Option(10, "--support-lookback", "-s", help="Number of days to look back for support levels"),
    output_dir: str = typer.Option("features", "--output-dir", "-o", help="Directory to save extracted features"),
) -> None:
    """
    Extract technical features from labeled pattern data.
    
    This command processes labeled pattern data and extracts technical
    features for machine learning model training.
    """
    try:
        _print_info(f"Extracting features from {labeled_data_path}")
        _print_info(f"Window size: {window_size}, Prior context: {prior_context_days} days")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting features...", total=None)
            
            results = build_features(
                labeled_data_path=labeled_data_path,
                window_size=window_size,
                prior_context_days=prior_context_days,
                support_lookback_days=support_lookback_days,
                output_dir=output_dir,
            )
            
            progress.update(task, completed=True)
        
        # Display results summary
        table = Table(title="Feature Extraction Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Features Extracted", str(results.get("features_count", 0)))
        table.add_row("Patterns Processed", str(results.get("patterns_count", 0)))
        table.add_row("Output Directory", str(results.get("output_dir", output_dir)))
        table.add_row("Feature Columns", str(len(results.get("feature_columns", []))))
        
        console.print(table)
        
        _print_success("Feature extraction completed successfully!")
        
    except FeatureExtractionError as e:
        _print_error(f"Feature extraction failed: {e}")
        sys.exit(1)
    except Exception as e:
        _print_error(f"Unexpected error: {e}")
        sys.exit(1)


@app.command()
def scan(
    tickers: List[str] = typer.Argument(..., help="List of stock ticker symbols to scan"),
    model_path: str = typer.Option(..., "--model", "-m", help="Path to the trained model file"),
    min_confidence: float = typer.Option(0.7, "--min-confidence", "-c", help="Minimum confidence for pattern detection"),
    window_size: int = typer.Option(30, "--window-size", "-w", help="Size of sliding window in days"),
    max_windows_per_ticker: int = typer.Option(10, "--max-windows", "-x", help="Maximum number of windows to evaluate per ticker"),
    save_results: bool = typer.Option(True, "--save", help="Whether to save results to file"),
    output_filename: Optional[str] = typer.Option(None, "--output", "-o", help="Custom output filename"),
) -> None:
    """
    Scan multiple stocks for trading patterns using a trained model.
    
    This command scans the specified stocks for trading patterns
    using a pre-trained machine learning model.
    """
    try:
        _print_info(f"Scanning {len(tickers)} tickers for patterns")
        _print_info(f"Model: {model_path}, Min confidence: {min_confidence}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning for patterns...", total=None)
            
            results = scan_patterns(
                tickers=tickers,
                model_path=model_path,
                min_confidence=min_confidence,
                window_size=window_size,
                max_windows_per_ticker=max_windows_per_ticker,
                save_results=save_results,
                output_filename=output_filename,
            )
            
            progress.update(task, completed=True)
        
        # Display results summary
        table = Table(title="Pattern Scanning Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Tickers Scanned", str(len(tickers)))
        table.add_row("Patterns Found", str(results.get("patterns_found", 0)))
        table.add_row("High Confidence", str(results.get("high_confidence_count", 0)))
        table.add_row("Average Confidence", f"{results.get('average_confidence', 0):.3f}")
        
        console.print(table)
        
        _print_success("Pattern scanning completed successfully!")
        
    except PatternScanningError as e:
        _print_error(f"Pattern scanning failed: {e}")
        sys.exit(1)
    except Exception as e:
        _print_error(f"Unexpected error: {e}")
        sys.exit(1)


@app.command()
def train(
    labeled_data_path: str = typer.Argument(..., help="Path to the labeled pattern data"),
    model_type: str = typer.Option("xgboost", "--model-type", "-t", help="Type of model to train (xgboost or random_forest)"),
    test_size: float = typer.Option(0.2, "--test-size", "-s", help="Fraction of data to use for testing"),
    use_cross_validation: bool = typer.Option(True, "--cross-validation", "-c", help="Whether to perform cross-validation"),
    output_dir: str = typer.Option("models", "--output-dir", "-o", help="Directory to save the trained model"),
) -> None:
    """
    Train a pattern detection model from labeled data.
    
    This command trains a machine learning model for pattern detection
    using the provided labeled data.
    """
    try:
        _print_info(f"Training {model_type} model from {labeled_data_path}")
        _print_info(f"Test size: {test_size}, Cross-validation: {use_cross_validation}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training model...", total=None)
            
            results = train_model(
                labeled_data_path=labeled_data_path,
                model_type=model_type,
                test_size=test_size,
                use_cross_validation=use_cross_validation,
                output_dir=output_dir,
            )
            
            progress.update(task, completed=True)
        
        # Display results summary
        table = Table(title="Model Training Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Model Type", model_type)
        table.add_row("Model Path", str(results.get("model_path", "N/A")))
        table.add_row("Training Accuracy", f"{results.get('training_metrics', {}).get('accuracy', 0):.3f}")
        table.add_row("Test Accuracy", f"{results.get('test_metrics', {}).get('accuracy', 0):.3f}")
        table.add_row("Training Date", str(results.get("training_date", "N/A")))
        
        console.print(table)
        
        _print_success("Model training completed successfully!")
        
    except ModelTrainingError as e:
        _print_error(f"Model training failed: {e}")
        sys.exit(1)
    except Exception as e:
        _print_error(f"Unexpected error: {e}")
        sys.exit(1)


@app.command()
def version() -> None:
    """Show the version of stock-analyzer."""
    from . import __version__
    console.print(f"stock-analyzer version {__version__}", style="bold blue")


if __name__ == "__main__":
    app() 