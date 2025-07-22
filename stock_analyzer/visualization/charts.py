"""
Pattern Visualizer for Stock Trading Patterns

This module provides optional visualization functionality to display labeled
stock patterns on candlestick charts using mplfinance.
"""

import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Handle imports for both direct execution and package usage
try:
    from ..patterns.labeler import PatternLabel
    from ..data.fetcher import fetch_hk_stocks, _load_cached_data
    from ..features.indicators import (
        find_support_resistance_levels, 
        simple_moving_average,
        price_volatility
    )
except ImportError:
    from stock_analyzer.patterns.labeler import PatternLabel
    from stock_analyzer.data.fetcher import fetch_hk_stocks, _load_cached_data
    from stock_analyzer.features.indicators import (
        find_support_resistance_levels, 
        simple_moving_average,
        price_volatility
    )

# Try to import mplfinance, handle gracefully if not available
try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False
    warnings.warn(
        "mplfinance not available. Install with: pip install mplfinance>=0.12.0"
    )


class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass


class MatchVisualizationError(VisualizationError):
    """Custom exception for pattern match visualization errors."""
    pass


@dataclass
class MatchRow:
    """
    Data class representing a single pattern match row from CSV.
    
    Attributes:
        ticker: Stock ticker symbol
        window_start_date: Start date of detection window
        window_end_date: End date of detection window
        confidence_score: ML model confidence score
        rank: Optional ranking of the match
        support_level: Optional precomputed support level
    """
    ticker: str
    window_start_date: str
    window_end_date: str
    confidence_score: float
    rank: Optional[int] = None
    support_level: Optional[float] = None


class PatternChartVisualizer:
    """
    Visualizer for displaying stock patterns on candlestick charts.
    """
    
    def __init__(self, charts_dir: str = "charts", require_mplfinance: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            charts_dir: Directory for saving chart images
            require_mplfinance: Whether to require mplfinance or allow fallback mode
        """
        if require_mplfinance and not MPLFINANCE_AVAILABLE:
            raise VisualizationError(
                "mplfinance is required for visualization. "
                "Install with: pip install mplfinance==0.12.10b0\n"
                "Or initialize with require_mplfinance=False for fallback mode"
            )
        
        self.charts_dir = charts_dir
        self.mplfinance_available = MPLFINANCE_AVAILABLE
        self._ensure_charts_directory()
    
    def _ensure_charts_directory(self) -> None:
        """Create charts directory if it doesn't exist."""
        if not os.path.exists(self.charts_dir):
            os.makedirs(self.charts_dir)
            print(f"✓ Created charts directory: {self.charts_dir}/")
    
    def load_matches_from_csv(self, csv_file_path: str) -> pd.DataFrame:
        """
        Load pattern matches from CSV file with validation.
        
        Args:
            csv_file_path: Path to matches CSV file
            
        Returns:
            pd.DataFrame: Validated matches DataFrame
            
        Raises:
            MatchVisualizationError: If file cannot be loaded or validated
        """
        try:
            if not os.path.exists(csv_file_path):
                raise MatchVisualizationError(f"Matches file not found: {csv_file_path}")
            
            matches_df = pd.read_csv(csv_file_path)
            
            # Validate required columns
            required_columns = ['ticker', 'window_start_date', 'window_end_date', 'confidence_score']
            missing_columns = set(required_columns) - set(matches_df.columns)
            
            if missing_columns:
                raise MatchVisualizationError(
                    f"Missing required columns in matches file: {missing_columns}"
                )
            
            # Validate data types and ranges
            if matches_df.empty:
                warnings.warn("Matches file is empty")
                return matches_df
            
            # Validate confidence scores
            if not (0 <= matches_df['confidence_score']).all() or not (matches_df['confidence_score'] <= 1).all():
                warnings.warn("Some confidence scores are outside the range [0, 1]")
            
            # Validate date formats
            for date_col in ['window_start_date', 'window_end_date']:
                try:
                    pd.to_datetime(matches_df[date_col])
                except ValueError as e:
                    raise MatchVisualizationError(f"Invalid date format in {date_col}: {e}")
            
            # Sort by confidence score descending
            matches_df = matches_df.sort_values('confidence_score', ascending=False).reset_index(drop=True)
            
            print(f"✓ Loaded {len(matches_df)} matches from {csv_file_path}")
            return matches_df
            
        except Exception as e:
            if isinstance(e, MatchVisualizationError):
                raise
            raise MatchVisualizationError(f"Error loading matches file: {e}")
    
    def validate_match_data(self, matches_df: pd.DataFrame) -> bool:
        """
        Validate matches DataFrame structure and content.
        
        Args:
            matches_df: DataFrame to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            MatchVisualizationError: If validation fails
        """
        if matches_df.empty:
            warnings.warn("Matches DataFrame is empty")
            return True
        
        # Check for null values in critical columns
        critical_columns = ['ticker', 'window_start_date', 'window_end_date', 'confidence_score']
        for col in critical_columns:
            if matches_df[col].isnull().any():
                raise MatchVisualizationError(f"Found null values in critical column: {col}")
        
        # Validate ticker formats (HK stocks should be XXXX.HK format)
        hk_ticker_pattern = r'^\d{4}\.HK$'
        invalid_tickers = []
        for ticker in matches_df['ticker'].unique():
            if not pd.Series([ticker]).str.match(hk_ticker_pattern).iloc[0]:
                invalid_tickers.append(ticker)
        
        if invalid_tickers:
            warnings.warn(f"Found non-HK ticker formats: {invalid_tickers}")
        
        # Validate date relationships
        for _, row in matches_df.iterrows():
            start_date = pd.to_datetime(row['window_start_date'])
            end_date = pd.to_datetime(row['window_end_date'])
            
            if start_date >= end_date:
                raise MatchVisualizationError(
                    f"Invalid date range for {row['ticker']}: start_date >= end_date"
                )
        
        return True
    
    def _calculate_support_level(self, data: pd.DataFrame, window_start: str, window_end: str) -> float:
        """
        Calculate support level for a given time window.
        
        Args:
            data: OHLCV DataFrame
            window_start: Start date of window
            window_end: End date of window
            
        Returns:
            float: Calculated support level
        """
        try:
            # Filter data to the specified window
            start_dt = pd.to_datetime(window_start)
            end_dt = pd.to_datetime(window_end)
            
            window_data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            
            if window_data.empty:
                warnings.warn(f"No data available for window {window_start} to {window_end}")
                return data['Low'].min()
            
            # Use technical indicators module to find support levels
            support_levels, _ = find_support_resistance_levels(window_data['Low'])
            
            if support_levels.notna().any():
                # Return the lowest support level within the window
                return support_levels.dropna().min()
            else:
                # Fallback to window minimum
                return window_data['Low'].min()
                
        except Exception as e:
            warnings.warn(f"Error calculating support level: {e}")
            return data['Low'].min()
    
    def _prepare_match_chart_data(self, 
                                 ticker: str, 
                                 window_start: str, 
                                 window_end: str,
                                 buffer_days: int = 10,
                                 context_days: int = 5) -> pd.DataFrame:
        """
        Prepare chart data for a pattern match with extended context.
        
        Args:
            ticker: Stock ticker
            window_start: Window start date
            window_end: Window end date  
            buffer_days: Days to add before window start
            context_days: Days to add after window end
            
        Returns:
            pd.DataFrame: Extended chart data
            
        Raises:
            MatchVisualizationError: If data preparation fails
        """
        try:
            # Calculate extended date range
            start_dt = pd.to_datetime(window_start) - timedelta(days=buffer_days)
            end_dt = pd.to_datetime(window_end) + timedelta(days=context_days)
            
            extended_start = start_dt.strftime("%Y-%m-%d")
            extended_end = end_dt.strftime("%Y-%m-%d")
            
            # Try to load from cache first
            cached_data = _load_cached_data(ticker)
            if cached_data is not None:
                # Filter cached data to required range
                filtered_data = cached_data[
                    (cached_data.index >= start_dt) & 
                    (cached_data.index <= end_dt)
                ]
                
                if len(filtered_data) > 0:
                    return filtered_data.sort_index()
            
            # Fallback to fetching new data
            data_dict = fetch_hk_stocks([ticker], extended_start, extended_end)
            
            if ticker not in data_dict or data_dict[ticker].empty:
                raise MatchVisualizationError(f"No data available for {ticker}")
            
            data = data_dict[ticker].copy()
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise MatchVisualizationError(f"Missing required columns: {missing_columns}")
            
            return data.sort_index()
            
        except Exception as e:
            if isinstance(e, MatchVisualizationError):
                raise
            raise MatchVisualizationError(f"Failed to prepare chart data for {ticker}: {e}")
    
    def visualize_pattern_match(self,
                               match_row: Union[pd.Series, MatchRow, Dict[str, Any]],
                               buffer_days: int = 10,
                               context_days: int = 5,
                               volume: bool = True,
                               show_support_level: bool = True,
                               figsize: Tuple[int, int] = (14, 10),
                               save: bool = False,
                               save_path: Optional[str] = None) -> None:
        """
        Visualize a single pattern match with overlays.
        
        Args:
            match_row: Pattern match data (Series, MatchRow, or dict)
            buffer_days: Days to show before detection window
            context_days: Days to show after detection window
            volume: Whether to show volume subplot
            show_support_level: Whether to calculate and show support level
            figsize: Figure size (width, height)
            save: Whether to save chart image
            save_path: Custom save path (optional)
            
        Raises:
            MatchVisualizationError: If visualization fails
        """
        import time
        start_time = time.time()
        
        try:
            # Extract match information
            if isinstance(match_row, pd.Series):
                ticker = match_row['ticker']
                window_start = match_row['window_start_date']
                window_end = match_row['window_end_date']
                confidence = match_row['confidence_score']
                rank = match_row.get('rank', None)
                precomputed_support = match_row.get('support_level', None)
            elif isinstance(match_row, MatchRow):
                ticker = match_row.ticker
                window_start = match_row.window_start_date
                window_end = match_row.window_end_date
                confidence = match_row.confidence_score
                rank = match_row.rank
                precomputed_support = match_row.support_level
            elif isinstance(match_row, dict):
                ticker = match_row['ticker']
                window_start = match_row['window_start_date']
                window_end = match_row['window_end_date']
                confidence = match_row['confidence_score']
                rank = match_row.get('rank', None)
                precomputed_support = match_row.get('support_level', None)
            else:
                raise MatchVisualizationError(f"Invalid match_row type: {type(match_row)}")
            
            print(f"📊 Visualizing pattern match for {ticker}")
            print(f"   Window: {window_start} to {window_end}")
            print(f"   Confidence: {confidence:.3f}")
            if rank is not None:
                print(f"   Rank: #{rank}")
            
            # Prepare chart data
            data = self._prepare_match_chart_data(
                ticker, window_start, window_end, buffer_days, context_days
            )
            
            if data.empty:
                raise MatchVisualizationError(f"No chart data available for {ticker}")
            
            # Calculate support level if needed
            support_level = None
            if show_support_level:
                if precomputed_support is not None:
                    support_level = precomputed_support
                else:
                    support_level = self._calculate_support_level(data, window_start, window_end)
            
            # Create the visualization
            self._create_match_chart(
                data=data,
                ticker=ticker,
                window_start=window_start,
                window_end=window_end,
                confidence=confidence,
                rank=rank,
                support_level=support_level,
                volume=volume,
                figsize=figsize
            )
            
            # Save chart if requested
            if save:
                save_filename = save_path or self._generate_match_save_path(ticker, window_start, confidence)
                plt.savefig(save_filename, dpi=300, bbox_inches='tight')
                print(f"💾 Chart saved to: {save_filename}")
            
            plt.show()
            
            # Performance check
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                warnings.warn(f"Chart generation took {elapsed_time:.2f}s (>1s target)")
            else:
                print(f"⚡ Chart generated in {elapsed_time:.3f}s")
                
        except Exception as e:
            if isinstance(e, MatchVisualizationError):
                raise
            raise MatchVisualizationError(f"Failed to visualize match: {e}")
    
    def _create_match_chart(self,
                           data: pd.DataFrame,
                           ticker: str,
                           window_start: str,
                           window_end: str,
                           confidence: float,
                           rank: Optional[int] = None,
                           support_level: Optional[float] = None,
                           volume: bool = True,
                           figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Create the actual pattern match chart with all overlays.
        
        Args:
            data: OHLCV data
            ticker: Stock ticker
            window_start: Detection window start date
            window_end: Detection window end date
            confidence: Model confidence score
            rank: Optional match rank
            support_level: Optional support level to display
            volume: Whether to include volume subplot
            figsize: Figure size
        """
        if self.mplfinance_available:
            self._create_mplfinance_chart(
                data, ticker, window_start, window_end, confidence,
                rank, support_level, volume, figsize
            )
        else:
            self._create_fallback_chart(
                data, ticker, window_start, window_end, confidence,
                rank, support_level, volume, figsize
            )
    
    def _create_mplfinance_chart(self,
                                data: pd.DataFrame,
                                ticker: str,
                                window_start: str,
                                window_end: str,
                                confidence: float,
                                rank: Optional[int] = None,
                                support_level: Optional[float] = None,
                                volume: bool = True,
                                figsize: Tuple[int, int] = (14, 10)) -> None:
        """Create chart using mplfinance for professional candlestick display."""
        # Prepare mplfinance style
        mc = mpf.make_marketcolors(
            up='g', down='r',
            edge='inherit',
            wick={'up': 'green', 'down': 'red'},
            volume='in'
        )
        
        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',
            y_on_right=True
        )
        
        # Create title
        title_parts = [f'{ticker} - Pattern Match Detection']
        if rank is not None:
            title_parts[0] += f' (Rank #{rank})'
        title_parts.append(f'Window: {window_start} to {window_end}')
        title_parts.append(f'Confidence: {confidence:.3f}')
        
        title = '\n'.join(title_parts)
        
        # Prepare additional plots for overlays
        additional_plots = []
        
        # Add support level horizontal line
        if support_level is not None:
            support_line = mpf.make_addplot(
                pd.Series([support_level] * len(data), index=data.index),
                color='orange',
                linestyle='--',
                width=2,
                alpha=0.7,
                secondary_y=False
            )
            additional_plots.append(support_line)
        
        # Create the main plot
        fig, axes = mpf.plot(
            data,
            type='candle',
            style=style,
            volume=volume,
            figsize=figsize,
            title=title,
            addplot=additional_plots if additional_plots else None,
            returnfig=True,
            warn_too_much_data=len(data)
        )
        
        # Add detection window overlay
        ax = axes[0]  # Main price axis
        
        # Convert dates to find indices for window highlighting
        start_dt = pd.to_datetime(window_start)
        end_dt = pd.to_datetime(window_end)
        
        # Add vertical lines for detection window boundaries
        ax.axvline(x=start_dt, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Detection Window Start')
        ax.axvline(x=end_dt, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Detection Window End')
        
        # Add shaded region for detection window
        ax.axvspan(start_dt, end_dt, alpha=0.15, color='blue', label='Detection Window')
        
        # Add support level annotation if present
        if support_level is not None:
            ax.text(0.02, 0.02, f'Support Level: ${support_level:.2f}', 
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8),
                   fontsize=10)
        
        # Add confidence score annotation
        ax.text(0.98, 0.98, f'Confidence: {confidence:.1%}', 
               transform=ax.transAxes,
               horizontalalignment='right',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               fontsize=12, fontweight='bold')
        
        # Add legend
        legend = ax.legend(loc='upper left')
        legend.set_alpha(0.8)
        
        plt.tight_layout()
    
    def _create_fallback_chart(self,
                              data: pd.DataFrame,
                              ticker: str,
                              window_start: str,
                              window_end: str,
                              confidence: float,
                              rank: Optional[int] = None,
                              support_level: Optional[float] = None,
                              volume: bool = True,
                              figsize: Tuple[int, int] = (14, 10)) -> None:
        """Create basic chart using matplotlib when mplfinance isn't available."""
        # Create subplots
        if volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        
        # Create title
        title_parts = [f'{ticker} - Pattern Match Detection (Basic Mode)']
        if rank is not None:
            title_parts[0] += f' (Rank #{rank})'
        title_parts.append(f'Window: {window_start} to {window_end}')
        title_parts.append(f'Confidence: {confidence:.3f}')
        
        title = '\n'.join(title_parts)
        fig.suptitle(title, fontsize=14)
        
        # Plot price data as line chart
        ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1.5)
        ax1.plot(data.index, data['High'], label='High', color='green', alpha=0.5, linewidth=0.5)
        ax1.plot(data.index, data['Low'], label='Low', color='red', alpha=0.5, linewidth=0.5)
        
        # Add detection window highlighting
        start_dt = pd.to_datetime(window_start)
        end_dt = pd.to_datetime(window_end)
        
        ax1.axvline(x=start_dt, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Window Start')
        ax1.axvline(x=end_dt, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Window End')
        ax1.axvspan(start_dt, end_dt, alpha=0.15, color='blue', label='Detection Window')
        
        # Add support level if available
        if support_level is not None:
            ax1.axhline(y=support_level, color='orange', linestyle='--', alpha=0.7, 
                       linewidth=2, label=f'Support: ${support_level:.2f}')
        
        # Add confidence annotation
        ax1.text(0.98, 0.98, f'Confidence: {confidence:.1%}', 
                transform=ax1.transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=12, fontweight='bold')
        
        ax1.set_ylabel('Price')
        legend1 = ax1.legend(loc='upper left')
        legend1.set_alpha(0.8)
        ax1.grid(True, alpha=0.3)
        
        # Plot volume if requested
        if volume and 'Volume' in data.columns:
            ax2.bar(data.index, data['Volume'], alpha=0.7, color='gray', label='Volume')
            ax2.axvspan(start_dt, end_dt, alpha=0.15, color='blue')
            ax2.set_ylabel('Volume')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add warning about fallback mode
        print("⚠️ Using fallback visualization mode (basic charts)")
        print("   Install mplfinance for professional candlestick charts: pip install mplfinance==0.12.10b0")
    
    def _generate_match_save_path(self, ticker: str, window_start: str, confidence: float) -> str:
        """
        Generate standardized save path for match charts.
        
        Args:
            ticker: Stock ticker
            window_start: Window start date
            confidence: Confidence score
            
        Returns:
            str: Generated save path
        """
        # Clean ticker for filename
        clean_ticker = ticker.replace('.', '_')
        
        # Format date for filename  
        date_str = window_start.replace('-', '')
        
        # Format confidence
        conf_str = f"{int(confidence * 100):03d}"
        
        filename = f"{clean_ticker}_{date_str}_conf{conf_str}.png"
        return os.path.join(self.charts_dir, filename)
    
    def visualize_all_matches(self,
                             matches_df: pd.DataFrame,
                             max_matches: int = 10,
                             min_confidence: float = 0.0,
                             save_all: bool = False,
                             **kwargs) -> None:
        """
        Visualize multiple pattern matches sequentially.
        
        Args:
            matches_df: DataFrame containing pattern matches
            max_matches: Maximum number of matches to visualize
            min_confidence: Minimum confidence threshold for visualization
            save_all: Whether to save all generated charts
            **kwargs: Additional arguments passed to visualize_pattern_match()
            
        Raises:
            MatchVisualizationError: If batch visualization fails
        """
        try:
            # Validate and filter matches
            self.validate_match_data(matches_df)
            
            if matches_df.empty:
                print("⚠️ No matches to visualize")
                return
            
            # Apply filters
            filtered_df = matches_df[
                matches_df['confidence_score'] >= min_confidence
            ].head(max_matches)
            
            if filtered_df.empty:
                print(f"⚠️ No matches found with confidence >= {min_confidence}")
                return
            
            print(f"🎯 Visualizing {len(filtered_df)} pattern matches")
            print(f"   Confidence threshold: {min_confidence:.3f}")
            print(f"   Max matches: {max_matches}")
            print("=" * 60)
            
            successful_visualizations = 0
            failed_visualizations = []
            
            for idx, (_, match_row) in enumerate(filtered_df.iterrows(), 1):
                try:
                    print(f"\n📊 Match {idx}/{len(filtered_df)}")
                    
                    # Set save flag for batch processing
                    kwargs_copy = kwargs.copy()
                    if save_all:
                        kwargs_copy['save'] = True
                    
                    self.visualize_pattern_match(match_row, **kwargs_copy)
                    successful_visualizations += 1
                    
                    # Add separator between charts
                    if idx < len(filtered_df):
                        print("-" * 40)
                    
                except Exception as e:
                    error_msg = f"{match_row['ticker']} ({match_row['window_start_date']}): {str(e)}"
                    failed_visualizations.append(error_msg)
                    warnings.warn(f"Failed to visualize match {idx}: {error_msg}")
                    continue
            
            # Summary
            print("\n" + "=" * 60)
            print(f"📈 Batch Visualization Summary:")
            print(f"   Successful: {successful_visualizations}")
            print(f"   Failed: {len(failed_visualizations)}")
            
            if failed_visualizations:
                print(f"   Failed matches:")
                for error in failed_visualizations[:5]:  # Show first 5 errors
                    print(f"     • {error}")
                if len(failed_visualizations) > 5:
                    print(f"     ... and {len(failed_visualizations) - 5} more")
            
        except Exception as e:
            raise MatchVisualizationError(f"Batch visualization failed: {e}")
    
    def visualize_matches_by_confidence(self,
                                       matches_df: pd.DataFrame,
                                       confidence_thresholds: List[float] = [0.9, 0.8, 0.7],
                                       max_per_threshold: int = 3,
                                       save_all: bool = False,
                                       **kwargs) -> Dict[float, int]:
        """
        Visualize matches grouped by confidence score thresholds.
        
        Args:
            matches_df: DataFrame containing pattern matches
            confidence_thresholds: List of confidence thresholds to group by
            max_per_threshold: Maximum matches per threshold group
            save_all: Whether to save all generated charts
            **kwargs: Additional arguments passed to visualize_pattern_match()
            
        Returns:
            Dict[float, int]: Number of matches visualized per threshold
            
        Raises:
            MatchVisualizationError: If visualization fails
        """
        try:
            self.validate_match_data(matches_df)
            
            if matches_df.empty:
                print("⚠️ No matches to visualize")
                return {}
            
            results = {}
            
            # Sort thresholds in descending order
            sorted_thresholds = sorted(confidence_thresholds, reverse=True)
            
            print(f"🎯 Visualizing matches by confidence thresholds")
            print(f"   Thresholds: {sorted_thresholds}")
            print(f"   Max per threshold: {max_per_threshold}")
            print("=" * 60)
            
            for threshold in sorted_thresholds:
                # Filter matches for this threshold
                threshold_matches = matches_df[
                    matches_df['confidence_score'] >= threshold
                ].head(max_per_threshold)
                
                if threshold_matches.empty:
                    print(f"\n🔍 Confidence >= {threshold:.3f}: No matches found")
                    results[threshold] = 0
                    continue
                
                print(f"\n🔍 Confidence >= {threshold:.3f}: {len(threshold_matches)} matches")
                print("-" * 40)
                
                successful_count = 0
                
                for idx, (_, match_row) in enumerate(threshold_matches.iterrows(), 1):
                    try:
                        print(f"   Match {idx}: {match_row['ticker']} - {match_row['confidence_score']:.3f}")
                        
                        kwargs_copy = kwargs.copy()
                        if save_all:
                            kwargs_copy['save'] = True
                        
                        self.visualize_pattern_match(match_row, **kwargs_copy)
                        successful_count += 1
                        
                    except Exception as e:
                        warnings.warn(f"Failed to visualize {match_row['ticker']}: {e}")
                        continue
                
                results[threshold] = successful_count
                print(f"   Visualized: {successful_count}/{len(threshold_matches)} matches")
            
            # Summary
            total_visualized = sum(results.values())
            print(f"\n📈 Confidence-Based Visualization Summary:")
            print(f"   Total matches visualized: {total_visualized}")
            for threshold, count in results.items():
                print(f"   Confidence >= {threshold:.3f}: {count} matches")
            
            return results
            
        except Exception as e:
            raise MatchVisualizationError(f"Confidence-based visualization failed: {e}")
    
    def generate_match_summary_report(self,
                                     matches_df: pd.DataFrame,
                                     save_report: bool = True,
                                     report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a summary report of pattern matches with key statistics.
        
        Args:
            matches_df: DataFrame containing pattern matches
            save_report: Whether to save report to file
            report_path: Custom path for report file
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        try:
            self.validate_match_data(matches_df)
            
            if matches_df.empty:
                print("⚠️ No matches to analyze")
                return {}
            
            # Calculate summary statistics
            summary = {
                'total_matches': len(matches_df),
                'unique_tickers': matches_df['ticker'].nunique(),
                'confidence_stats': {
                    'mean': matches_df['confidence_score'].mean(),
                    'median': matches_df['confidence_score'].median(),
                    'std': matches_df['confidence_score'].std(),
                    'min': matches_df['confidence_score'].min(),
                    'max': matches_df['confidence_score'].max()
                },
                'confidence_distribution': {
                    'high_confidence_0_9': len(matches_df[matches_df['confidence_score'] >= 0.9]),
                    'medium_confidence_0_7_0_9': len(matches_df[
                        (matches_df['confidence_score'] >= 0.7) & 
                        (matches_df['confidence_score'] < 0.9)
                    ]),
                    'low_confidence_below_0_7': len(matches_df[matches_df['confidence_score'] < 0.7])
                },
                'top_tickers': matches_df['ticker'].value_counts().head(10).to_dict(),
                'date_range': {
                    'earliest_window': matches_df['window_start_date'].min(),
                    'latest_window': matches_df['window_end_date'].max()
                }
            }
            
            # Print summary
            print("📊 Pattern Match Summary Report")
            print("=" * 50)
            print(f"Total Matches: {summary['total_matches']}")
            print(f"Unique Tickers: {summary['unique_tickers']}")
            print(f"Date Range: {summary['date_range']['earliest_window']} to {summary['date_range']['latest_window']}")
            
            print(f"\nConfidence Score Statistics:")
            conf_stats = summary['confidence_stats']
            print(f"  Mean: {conf_stats['mean']:.3f}")
            print(f"  Median: {conf_stats['median']:.3f}")
            print(f"  Std Dev: {conf_stats['std']:.3f}")
            print(f"  Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")
            
            print(f"\nConfidence Distribution:")
            conf_dist = summary['confidence_distribution']
            print(f"  High (≥0.9): {conf_dist['high_confidence_0_9']}")
            print(f"  Medium (0.7-0.9): {conf_dist['medium_confidence_0_7_0_9']}")
            print(f"  Low (<0.7): {conf_dist['low_confidence_below_0_7']}")
            
            print(f"\nTop Tickers by Match Count:")
            for ticker, count in list(summary['top_tickers'].items())[:5]:
                print(f"  {ticker}: {count} matches")
            
            # Save report if requested
            if save_report:
                if report_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_path = os.path.join(self.charts_dir, f"match_summary_{timestamp}.txt")
                
                with open(report_path, 'w') as f:
                    f.write("Pattern Match Summary Report\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write(f"Total Matches: {summary['total_matches']}\n")
                    f.write(f"Unique Tickers: {summary['unique_tickers']}\n")
                    f.write(f"Date Range: {summary['date_range']['earliest_window']} to {summary['date_range']['latest_window']}\n\n")
                    
                    f.write("Confidence Score Statistics:\n")
                    for key, value in conf_stats.items():
                        f.write(f"  {key.title()}: {value:.3f}\n")
                    
                    f.write("\nConfidence Distribution:\n")
                    for key, value in conf_dist.items():
                        f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
                    
                    f.write("\nTop Tickers by Match Count:\n")
                    for ticker, count in summary['top_tickers'].items():
                        f.write(f"  {ticker}: {count} matches\n")
                
                print(f"💾 Summary report saved to: {report_path}")
            
            return summary
            
        except Exception as e:
            raise MatchVisualizationError(f"Failed to generate summary report: {e}")
    
    def _get_extended_date_range(self, 
                                start_date: str, 
                                end_date: str, 
                                buffer_days: int = 30) -> Tuple[str, str]:
        """
        Get extended date range for better chart context.
        
        Args:
            start_date: Pattern start date
            end_date: Pattern end date
            buffer_days: Days to add before and after pattern
            
        Returns:
            Tuple[str, str]: Extended start and end dates
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        extended_start = (start_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        extended_end = (end_dt + timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        
        return extended_start, extended_end
    
    def _prepare_chart_data(self, 
                           ticker: str, 
                           start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """
        Prepare chart data for visualization.
        
        Args:
            ticker: Stock ticker
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            pd.DataFrame: Prepared chart data
            
        Raises:
            VisualizationError: If data preparation fails
        """
        try:
            # Fetch data using existing data fetcher
            data_dict = fetch_hk_stocks([ticker], start_date, end_date)
            
            if ticker not in data_dict or data_dict[ticker].empty:
                raise VisualizationError(f"No data available for {ticker}")
            
            data = data_dict[ticker].copy()
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise VisualizationError(f"Missing required columns: {missing_columns}")
            
            # Sort by date
            data = data.sort_index()
            
            return data
            
        except Exception as e:
            raise VisualizationError(f"Failed to prepare chart data: {e}")
    
    def _create_pattern_highlight(self, 
                                 data: pd.DataFrame,
                                 pattern_start: str,
                                 pattern_end: str,
                                 label_type: str = "positive") -> List[Dict[str, Any]]:
        """
        Create pattern highlight annotations for mplfinance.
        
        Args:
            data: Chart data
            pattern_start: Pattern start date
            pattern_end: Pattern end date
            label_type: Type of pattern label
            
        Returns:
            List[Dict]: mplfinance annotation objects
        """
        # Color mapping for different label types
        color_map = {
            "positive": "green",
            "negative": "red", 
            "neutral": "orange"
        }
        
        color = color_map.get(label_type, "blue")
        
        # Find date indices in the data
        start_dt = pd.Timestamp(pattern_start)
        end_dt = pd.Timestamp(pattern_end)
        
        # Filter data to pattern range
        pattern_data = data[(data.index >= start_dt) & (data.index <= end_dt)]
        
        if pattern_data.empty:
            warnings.warn(f"No data found in pattern range {pattern_start} to {pattern_end}")
            return []
        
        # Create vertical lines for pattern boundaries
        annotations = []
        
        # Start line
        annotations.append(
            dict(
                type='line',
                x0=start_dt, x1=start_dt,
                y0=pattern_data['Low'].min() * 0.98,
                y1=pattern_data['High'].max() * 1.02,
                line=dict(color=color, width=2, dash='dash'),
                name=f'Pattern Start ({label_type})'
            )
        )
        
        # End line
        annotations.append(
            dict(
                type='line',
                x0=end_dt, x1=end_dt,
                y0=pattern_data['Low'].min() * 0.98,
                y1=pattern_data['High'].max() * 1.02,
                line=dict(color=color, width=2, dash='dash'),
                name=f'Pattern End ({label_type})'
            )
        )
        
        return annotations
    
    def display_labeled_pattern(self,
                               label: PatternLabel,
                               buffer_days: int = 30,
                               chart_style: str = 'charles',
                               volume: bool = True,
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None) -> None:
        """
        Display a labeled pattern on a candlestick chart.
        
        Args:
            label: PatternLabel instance to display
            buffer_days: Days to show before/after pattern
            chart_style: mplfinance chart style
            volume: Whether to show volume subplot
            figsize: Figure size (width, height)
            save_path: Optional path to save chart image
            
        Raises:
            VisualizationError: If visualization fails
        """
        # Get extended date range for context
        extended_start, extended_end = self._get_extended_date_range(
            label.start_date, label.end_date, buffer_days
        )
        
        # Prepare chart data
        data = self._prepare_chart_data(label.ticker, extended_start, extended_end)
        
        # Create pattern highlight
        pattern_highlights = self._create_pattern_highlight(
            data, label.start_date, label.end_date, label.label_type
        )
        
        # Prepare mplfinance style
        mc = mpf.make_marketcolors(
            up='g', down='r',
            edge='inherit',
            wick={'up': 'green', 'down': 'red'},
            volume='in'
        )
        
        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',
            y_on_right=True
        )
        
        # Create the plot
        fig, axes = mpf.plot(
            data,
            type='candle',
            style=style,
            volume=volume,
            figsize=figsize,
            title=f'{label.ticker} - Pattern: {label.label_type.capitalize()}\n'
                  f'Period: {label.start_date} to {label.end_date}',
            returnfig=True,
            warn_too_much_data=len(data)
        )
        
        # Add pattern boundaries as vertical lines
        ax = axes[0]  # Main price axis
        
        # Convert dates to matplotlib dates for plotting
        start_dt = pd.Timestamp(label.start_date)
        end_dt = pd.Timestamp(label.end_date)
        
        # Color mapping
        color_map = {
            "positive": "green",
            "negative": "red", 
            "neutral": "orange"
        }
        color = color_map.get(label.label_type, "blue")
        
        # Add vertical lines for pattern boundaries
        ax.axvline(x=start_dt, color=color, linestyle='--', alpha=0.7, linewidth=2, label='Pattern Start')
        ax.axvline(x=end_dt, color=color, linestyle='--', alpha=0.7, linewidth=2, label='Pattern End')
        
        # Add shaded region for pattern
        ax.axvspan(start_dt, end_dt, alpha=0.1, color=color, label=f'Pattern ({label.label_type})')
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Add notes if available
        if label.notes:
            ax.text(0.02, 0.98, f"Notes: {label.notes}", 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def compare_patterns(self,
                        labels: List[PatternLabel],
                        buffer_days: int = 30,
                        figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Compare multiple labeled patterns side by side.
        
        Args:
            labels: List of PatternLabel instances to compare
            buffer_days: Days to show before/after each pattern
            figsize: Figure size (width, height)
            
        Raises:
            VisualizationError: If visualization fails
        """
        if not labels:
            raise VisualizationError("No patterns provided for comparison")
        
        if len(labels) > 4:
            warnings.warn("Displaying more than 4 patterns may result in crowded charts")
        
        # Calculate subplot layout
        n_patterns = len(labels)
        n_cols = min(2, n_patterns)
        n_rows = (n_patterns + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_patterns == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, label in enumerate(labels):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                ax = axes[row][col]
            else:
                ax = axes[col]
            
            try:
                # Get extended date range
                extended_start, extended_end = self._get_extended_date_range(
                    label.start_date, label.end_date, buffer_days
                )
                
                # Prepare chart data
                data = self._prepare_chart_data(label.ticker, extended_start, extended_end)
                
                # Simple OHLC plot using matplotlib
                self._plot_simple_candlestick(ax, data, label)
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading {label.ticker}:\n{str(e)}", 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f"{label.ticker} - Error")
        
        # Hide unused subplots
        for i in range(n_patterns, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                ax = axes[row][col]
            else:
                ax = axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_simple_candlestick(self, ax, data: pd.DataFrame, label: PatternLabel) -> None:
        """
        Plot a simple candlestick chart on given axis.
        
        Args:
            ax: Matplotlib axis
            data: OHLC data
            label: Pattern label
        """
        # Color mapping
        color_map = {
            "positive": "green",
            "negative": "red", 
            "neutral": "orange"
        }
        color = color_map.get(label.label_type, "blue")
        
        # Plot candlesticks
        for idx, (date, row) in enumerate(data.iterrows()):
            # Determine candle color
            candle_color = 'green' if row['Close'] > row['Open'] else 'red'
            
            # Plot the wick (high-low line)
            ax.plot([idx, idx], [row['Low'], row['High']], 
                   color='black', linewidth=1)
            
            # Plot the body (open-close rectangle)
            body_height = abs(row['Close'] - row['Open'])
            body_bottom = min(row['Open'], row['Close'])
            
            rect = Rectangle((idx - 0.3, body_bottom), 0.6, body_height,
                           facecolor=candle_color, alpha=0.7)
            ax.add_patch(rect)
        
        # Add pattern boundaries
        start_dt = pd.Timestamp(label.start_date)
        end_dt = pd.Timestamp(label.end_date)
        
        # Find indices for pattern boundaries
        start_idx = None
        end_idx = None
        
        for idx, date in enumerate(data.index):
            if date >= start_dt and start_idx is None:
                start_idx = idx
            if date <= end_dt:
                end_idx = idx
        
        if start_idx is not None and end_idx is not None:
            ax.axvline(x=start_idx, color=color, linestyle='--', alpha=0.7, linewidth=2)
            ax.axvline(x=end_idx, color=color, linestyle='--', alpha=0.7, linewidth=2)
            ax.axvspan(start_idx, end_idx, alpha=0.1, color=color)
        
        # Format the plot
        ax.set_title(f"{label.ticker} - {label.label_type.capitalize()}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        
        # Set x-axis labels to show dates
        n_ticks = min(10, len(data))
        tick_indices = [i * len(data) // n_ticks for i in range(n_ticks)]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([data.index[i].strftime('%Y-%m-%d') for i in tick_indices], 
                          rotation=45)


# Convenience functions for notebook usage
def display_labeled_pattern(label: PatternLabel, **kwargs) -> None:
    """
    Display a labeled pattern on a candlestick chart.
    
    Args:
        label: PatternLabel instance to display
        **kwargs: Additional arguments passed to PatternChartVisualizer.display_labeled_pattern()
    
    Raises:
        VisualizationError: If visualization fails
    """
    visualizer = PatternChartVisualizer()
    visualizer.display_labeled_pattern(label, **kwargs)


def compare_patterns(labels: List[PatternLabel], **kwargs) -> None:
    """
    Compare multiple labeled patterns side by side.
    
    Args:
        labels: List of PatternLabel instances to compare
        **kwargs: Additional arguments passed to PatternChartVisualizer.compare_patterns()
    
    Raises:
        VisualizationError: If visualization fails
    """
    visualizer = PatternChartVisualizer()
    visualizer.compare_patterns(labels, **kwargs)


# Convenience functions for pattern match visualization
def visualize_match(match_row: Union[pd.Series, MatchRow, Dict[str, Any]], **kwargs) -> None:
    """
    Visualize a single pattern match with overlays.
    
    Args:
        match_row: Pattern match data (Series, MatchRow, or dict)
        **kwargs: Additional arguments passed to PatternChartVisualizer.visualize_pattern_match()
    
    Raises:
        MatchVisualizationError: If visualization fails
    """
    visualizer = PatternChartVisualizer()
    visualizer.visualize_pattern_match(match_row, **kwargs)


def visualize_matches_from_csv(csv_file_path: str, 
                              max_matches: int = 5,
                              min_confidence: float = 0.7,
                              **kwargs) -> None:
    """
    Load and visualize pattern matches from CSV file.
    
    Args:
        csv_file_path: Path to matches CSV file
        max_matches: Maximum number of matches to visualize
        min_confidence: Minimum confidence threshold
        **kwargs: Additional arguments passed to visualization methods
    
    Raises:
        MatchVisualizationError: If loading or visualization fails
    """
    visualizer = PatternChartVisualizer()
    matches_df = visualizer.load_matches_from_csv(csv_file_path)
    visualizer.visualize_all_matches(
        matches_df, 
        max_matches=max_matches, 
        min_confidence=min_confidence,
        **kwargs
    )


def plot_match(ticker: str, 
               window_start: str, 
               window_end: str, 
               confidence_score: float,
               **kwargs) -> None:
    """
    Quick plot function for a single match by parameters.
    
    Args:
        ticker: Stock ticker symbol
        window_start: Detection window start date
        window_end: Detection window end date  
        confidence_score: ML model confidence score
        **kwargs: Additional arguments passed to visualize_pattern_match()
    
    Raises:
        MatchVisualizationError: If visualization fails
    """
    match_data = {
        'ticker': ticker,
        'window_start_date': window_start,
        'window_end_date': window_end,
        'confidence_score': confidence_score
    }
    
    visualizer = PatternChartVisualizer()
    visualizer.visualize_pattern_match(match_data, **kwargs)


def analyze_matches_by_confidence(matches_df: pd.DataFrame,
                                 thresholds: List[float] = [0.9, 0.8, 0.7],
                                 **kwargs) -> Dict[float, int]:
    """
    Analyze and visualize matches grouped by confidence thresholds.
    
    Args:
        matches_df: DataFrame containing pattern matches
        thresholds: List of confidence thresholds to analyze
        **kwargs: Additional arguments passed to visualization methods
    
    Returns:
        Dict[float, int]: Number of matches visualized per threshold
    
    Raises:
        MatchVisualizationError: If analysis fails
    """
    visualizer = PatternChartVisualizer()
    return visualizer.visualize_matches_by_confidence(
        matches_df, 
        confidence_thresholds=thresholds,
        **kwargs
    )


def generate_matches_report(matches_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Generate a comprehensive summary report of pattern matches.
    
    Args:
        matches_df: DataFrame containing pattern matches
        **kwargs: Additional arguments passed to report generation
    
    Returns:
        Dict[str, Any]: Summary statistics and report data
    
    Raises:
        MatchVisualizationError: If report generation fails
    """
    visualizer = PatternChartVisualizer()
    return visualizer.generate_match_summary_report(matches_df, **kwargs) 