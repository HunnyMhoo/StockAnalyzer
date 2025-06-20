"""
Pattern Visualizer for Stock Trading Patterns

This module provides optional visualization functionality to display labeled
stock patterns on candlestick charts using mplfinance.
"""

import warnings
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Handle imports for both direct execution and package usage
try:
    from .pattern_labeler import PatternLabel
    from .data_fetcher import fetch_hk_stocks
except ImportError:
    from pattern_labeler import PatternLabel
    from data_fetcher import fetch_hk_stocks

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


class PatternChartVisualizer:
    """
    Visualizer for displaying stock patterns on candlestick charts.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        if not MPLFINANCE_AVAILABLE:
            raise VisualizationError(
                "mplfinance is required for visualization. "
                "Install with: pip install mplfinance>=0.12.0"
            )
    
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