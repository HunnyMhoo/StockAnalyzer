# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all,-language_info,-toc,-latex_envs
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [raw]
# # Pattern Explorer & Strategy Back-tester
# **Interactive Pattern Discovery and Historical Examples**
#
# This notebook provides an interactive interface for:
# - Finding real historical examples of any registered chart pattern
# - Exploring pattern confidence distributions
# - Visualizing pattern examples with technical indicators
# - Quick strategy backtesting and performance analysis
#
# **Target Users:** Quant Analysts and Portfolio Managers
# **Performance Goal:** Example grid renders ≥50 hits in < 1 second

# %%
# Import required libraries
import sys
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Interactive widgets
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import bqplot as bq
from bqplot import pyplot as bqplt

# Add project root to path
project_root = Path().resolve()
while not (project_root / 'stock_analyzer').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

try:
    # Stock analyzer imports
    from stock_analyzer.data.signal_store import SignalStore, query_signals
    from stock_analyzer.analysis.strategy import StrategyManager, StrategyConfig
    from stock_analyzer.analysis.backtester import BacktestEngine, quick_backtest
    from stock_analyzer.visualization.charts import PatternMatchVisualizer
    from stock_analyzer.config import settings
    
    print("✅ Successfully imported stock_analyzer modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")

print(f"📊 Pattern Explorer initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
# Configuration and Setup
class PatternExplorerConfig:
    """Configuration for Pattern Explorer interface."""
    
    def __init__(self):
        # Interface settings
        self.max_examples_display = 50
        self.chart_width = 800
        self.chart_height = 400
        self.grid_columns = 3
        
        # Data settings
        self.signals_dir = "signals"
        self.default_days_lookback = 365
        self.min_confidence_default = 0.70
        
        # Performance settings
        self.enable_caching = True
        self.max_cache_size = 100

# Initialize configuration
config = PatternExplorerConfig()

# Initialize signal store
try:
    signal_store = SignalStore(base_dir=config.signals_dir)
    print(f"✅ Signal store initialized: {config.signals_dir}")
    
    # Check available data
    patterns = signal_store.get_available_patterns()
    date_range = signal_store.get_date_range()
    
    if patterns:
        print(f"📈 Available patterns: {', '.join(patterns)}")
    else:
        print("⚠️  No pattern data found. Please run pattern scanning first.")
        
    if date_range[0] and date_range[1]:
        print(f"📅 Data range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
    else:
        print("⚠️  No date range information available")
        
except Exception as e:
    print(f"❌ Signal store initialization failed: {e}")
    signal_store = None

# %%
# Interactive Pattern Explorer Interface
class PatternExplorer:
    """Interactive pattern exploration interface with widgets."""
    
    def __init__(self, signal_store: SignalStore, config: PatternExplorerConfig):
        """Initialize pattern explorer with signal store and configuration."""
        self.signal_store = signal_store
        self.config = config
        self.cached_results = {}
        
        # Get available data
        self.available_patterns = signal_store.get_available_patterns() if signal_store else []
        self.date_range = signal_store.get_date_range() if signal_store else (None, None)
        
        # Initialize widgets
        self._create_widgets()
        self._setup_interactions()
        
        # Results storage
        self.current_results = None
        self.current_visualizations = []
    
    def _create_widgets(self):
        """Create interactive widgets for pattern exploration."""
        
        # Pattern selection
        pattern_options = ['All Patterns'] + self.available_patterns
        self.pattern_dropdown = widgets.Dropdown(
            options=pattern_options,
            value=pattern_options[0] if pattern_options else 'All Patterns',
            description='Pattern:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        # Date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.config.default_days_lookback)
        
        if self.date_range[0] and self.date_range[1]:
            start_date = self.date_range[0].date()
            end_date = self.date_range[1].date()
        
        self.start_date_picker = widgets.DatePicker(
            description='Start Date:',
            value=start_date,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        self.end_date_picker = widgets.DatePicker(
            description='End Date:',
            value=end_date,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        # Confidence slider
        self.confidence_slider = widgets.FloatSlider(
            value=self.config.min_confidence_default,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Min Confidence:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'),
            readout_format='.2f'
        )
        
        # Limit slider
        self.limit_slider = widgets.IntSlider(
            value=25,
            min=5,
            max=self.config.max_examples_display,
            step=5,
            description='Max Examples:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Ticker filter
        self.ticker_input = widgets.Text(
            value='',
            placeholder='e.g., 0700.HK, 0388.HK (optional)',
            description='Tickers:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Search button
        self.search_button = widgets.Button(
            description='🔍 Find Examples',
            button_style='primary',
            layout=widgets.Layout(width='150px', height='35px')
        )
        
        # Results info
        self.results_info = widgets.HTML(
            value="<i>Click 'Find Examples' to search for patterns...</i>",
            layout=widgets.Layout(margin='10px 0px')
        )
        
        # Output area for results
        self.output_area = widgets.Output(layout=widgets.Layout(width='100%'))
        
        # Quick stats
        self.stats_output = widgets.Output()
    
    def _setup_interactions(self):
        """Setup widget interactions and event handlers."""
        self.search_button.on_click(self._on_search_clicked)
        
        # Auto-update on parameter changes (debounced)
        self.confidence_slider.observe(self._on_parameter_changed, names='value')
        self.limit_slider.observe(self._on_parameter_changed, names='value')
    
    def _on_parameter_changed(self, change):
        """Handle parameter changes with auto-refresh."""
        if self.current_results is not None:
            # Auto-refresh results with new parameters
            self._update_display()
    
    def _on_search_clicked(self, button):
        """Handle search button click."""
        self._search_patterns()
    
    def _search_patterns(self):
        """Execute pattern search with current parameters."""
        try:
            with self.output_area:
                clear_output(wait=True)
                print("🔍 Searching for patterns...")
            
            # Get search parameters
            pattern_id = None if self.pattern_dropdown.value == 'All Patterns' else self.pattern_dropdown.value
            start_date = datetime.combine(self.start_date_picker.value, datetime.min.time())
            end_date = datetime.combine(self.end_date_picker.value, datetime.min.time())
            min_confidence = self.confidence_slider.value
            limit = self.limit_slider.value
            
            # Parse tickers
            tickers = None
            if self.ticker_input.value.strip():
                tickers = [t.strip().upper() for t in self.ticker_input.value.split(',') if t.strip()]
            
            # Execute search
            search_start = datetime.now()
            query_result = self.signal_store.read_signals(
                start_date=start_date,
                end_date=end_date,
                pattern_id=pattern_id,
                min_confidence=min_confidence,
                tickers=tickers,
                limit=limit
            )
            search_time = (datetime.now() - search_start).total_seconds()
            
            # Store results
            self.current_results = query_result
            
            # Update display
            self._update_display()
            
            # Update info
            self.results_info.value = f"""
            <div style='background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <b>Search Results:</b> Found {query_result.total_records} examples in {search_time:.2f}s<br>
                <b>Patterns:</b> {', '.join(query_result.patterns_found) if query_result.patterns_found else 'None'}<br>
                <b>Tickers:</b> {len(query_result.tickers_found)} unique symbols<br>
                <b>Date Range:</b> {query_result.date_range[0].strftime('%Y-%m-%d')} to {query_result.date_range[1].strftime('%Y-%m-%d')}
            </div>
            """
            
        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                print(f"❌ Search failed: {e}")
            
            self.results_info.value = f"<div style='color: red;'>Search failed: {e}</div>"
    
    def _update_display(self):
        """Update the results display with current data."""
        if self.current_results is None or self.current_results.signals_df.empty:
            with self.output_area:
                clear_output(wait=True)
                print("No results to display")
            return
        
        try:
            # Filter results by current parameters
            filtered_df = self.current_results.signals_df[
                self.current_results.signals_df['confidence'] >= self.confidence_slider.value
            ].head(self.limit_slider.value)
            
            with self.output_area:
                clear_output(wait=True)
                
                if filtered_df.empty:
                    print("No examples match current filters")
                    return
                
                # Display summary statistics
                print(f"📊 Pattern Examples Summary ({len(filtered_df)} examples)")
                print("=" * 60)
                
                # Confidence distribution
                conf_stats = filtered_df['confidence'].describe()
                print(f"Confidence - Mean: {conf_stats['mean']:.3f}, Std: {conf_stats['std']:.3f}")
                print(f"           - Range: {conf_stats['min']:.3f} to {conf_stats['max']:.3f}")
                
                # Pattern breakdown
                if 'pattern_id' in filtered_df.columns:
                    pattern_counts = filtered_df['pattern_id'].value_counts()
                    print(f"\nPattern Distribution:")
                    for pattern, count in pattern_counts.head(5).items():
                        print(f"  • {pattern}: {count} examples")
                
                # Top tickers
                ticker_counts = filtered_df['ticker'].value_counts()
                print(f"\nTop Tickers:")
                for ticker, count in ticker_counts.head(5).items():
                    print(f"  • {ticker}: {count} examples")
                
                # Recent examples
                print(f"\n🔍 Recent Examples (Top {min(10, len(filtered_df))}):")
                print("-" * 60)
                
                display_df = filtered_df.head(10)[['ticker', 'pattern_id', 'confidence', 'date']].copy()
                display_df['confidence'] = display_df['confidence'].round(3)
                display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
                
                # Display as formatted table
                for idx, row in display_df.iterrows():
                    print(f"{row['ticker']:>8} | {row['pattern_id']:>12} | {row['confidence']:>6.3f} | {row['date']}")
                
                # Show visualization options
                print(f"\n📈 Visualization Options:")
                print("• Use the chart widgets below to visualize specific examples")
                print("• Copy ticker symbols to pattern visualizer for detailed charts")
                
            # Update quick stats
            with self.stats_output:
                clear_output(wait=True)
                self._display_confidence_distribution(filtered_df)
                
        except Exception as e:
            with self.output_area:
                clear_output(wait=True)
                print(f"❌ Display update failed: {e}")
    
    def _display_confidence_distribution(self, df):
        """Display confidence distribution chart."""
        if df.empty:
            return
            
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Confidence histogram
            ax1.hist(df['confidence'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Confidence Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Pattern distribution (if multiple patterns)
            if 'pattern_id' in df.columns and df['pattern_id'].nunique() > 1:
                pattern_counts = df['pattern_id'].value_counts()
                ax2.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%')
                ax2.set_title('Pattern Distribution')
            else:
                # Timeline of signals
                df_time = df.copy()
                df_time['date'] = pd.to_datetime(df_time['date'])
                df_time_grouped = df_time.groupby(df_time['date'].dt.date).size()
                
                ax2.plot(df_time_grouped.index, df_time_grouped.values, marker='o', linewidth=2)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Signal Count')
                ax2.set_title('Signals Over Time')
                ax2.tick_params(axis='x', rotation=45)
                
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Chart display failed: {e}")
    
    def display_interface(self):
        """Display the complete pattern explorer interface."""
        
        # Header
        header = widgets.HTML(
            value="""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='margin: 0; font-size: 24px;'>📊 Pattern Explorer</h2>
                <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>
                    Find and analyze historical chart pattern examples
                </p>
            </div>
            """
        )
        
        # Control panel
        controls_box = widgets.VBox([
            widgets.HTML("<h3>🔧 Search Parameters</h3>"),
            widgets.HBox([
                self.pattern_dropdown,
                self.start_date_picker,
                self.end_date_picker
            ]),
            widgets.HBox([
                self.confidence_slider,
                self.limit_slider
            ]),
            widgets.HBox([
                self.ticker_input,
                self.search_button
            ]),
            self.results_info
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            padding='15px',
            border_radius='5px',
            margin='10px 0px'
        ))
        
        # Results area
        results_box = widgets.VBox([
            widgets.HTML("<h3>📈 Search Results</h3>"),
            self.output_area,
            widgets.HTML("<h3>📊 Statistics</h3>"),
            self.stats_output
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            padding='15px',
            border_radius='5px',
            margin='10px 0px'
        ))
        
        # Display complete interface
        display(widgets.VBox([
            header,
            controls_box,
            results_box
        ]))

# %%
# Initialize and Display Pattern Explorer
if signal_store:
    explorer = PatternExplorer(signal_store, config)
    explorer.display_interface()
else:
    display(widgets.HTML("""
    <div style='background: #ffe6e6; border: 1px solid #ff9999; padding: 20px; border-radius: 5px;'>
        <h3 style='color: #cc0000; margin-top: 0;'>⚠️ Signal Store Not Available</h3>
        <p>The pattern explorer requires access to the signal store. Please ensure:</p>
        <ul>
            <li>Pattern scanning has been run to generate signals</li>
            <li>Signal files exist in the signals directory</li>
            <li>Proper file permissions are set</li>
        </ul>
        <p><strong>Next Steps:</strong> Run the pattern detection notebook first to generate signal data.</p>
    </div>
    """))

# %%
# Quick Strategy Backtesting Interface
class QuickBacktester:
    """Quick strategy backtesting interface."""
    
    def __init__(self, signal_store: SignalStore):
        """Initialize quick backtester."""
        self.signal_store = signal_store
        self.strategy_manager = StrategyManager()
        
        # Get available patterns for strategy selection
        self.available_patterns = signal_store.get_available_patterns() if signal_store else []
        
        self._create_widgets()
        self._setup_interactions()
        
        self.backtest_results = None
    
    def _create_widgets(self):
        """Create backtesting interface widgets."""
        
        # Strategy selection
        self.strategy_dropdown = widgets.Dropdown(
            options=['conservative', 'aggressive', 'trend_following', 'scalping'],
            value='conservative',
            description='Strategy Template:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        
        # Pattern selection
        pattern_options = self.available_patterns if self.available_patterns else ['bear_flag']
        self.pattern_dropdown = widgets.Dropdown(
            options=pattern_options,
            value=pattern_options[0],
            description='Pattern:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        # Capital input
        self.capital_input = widgets.FloatText(
            value=100000.0,
            description='Initial Capital:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        # Confidence override
        self.confidence_input = widgets.FloatSlider(
            value=0.75,
            min=0.5,
            max=0.95,
            step=0.05,
            description='Min Confidence:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'),
            readout_format='.2f'
        )
        
        # Backtest button
        self.backtest_button = widgets.Button(
            description='🚀 Run Backtest',
            button_style='success',
            layout=widgets.Layout(width='150px', height='35px')
        )
        
        # Results output
        self.backtest_output = widgets.Output()
        
        # Results info
        self.backtest_info = widgets.HTML(
            value="<i>Configure parameters and click 'Run Backtest' to start...</i>"
        )
    
    def _setup_interactions(self):
        """Setup widget interactions."""
        self.backtest_button.on_click(self._on_backtest_clicked)
    
    def _on_backtest_clicked(self, button):
        """Handle backtest button click."""
        self._run_backtest()
    
    def _run_backtest(self):
        """Execute strategy backtest."""
        try:
            with self.backtest_output:
                clear_output(wait=True)
                print("🚀 Starting backtest...")
            
            # Create strategy from template
            strategy = self.strategy_manager.create_strategy(
                template_name=self.strategy_dropdown.value,
                pattern_id=self.pattern_dropdown.value,
                min_confidence=self.confidence_input.value
            )
            
            # Get signals for the pattern
            signals_result = self.signal_store.read_signals(
                pattern_id=self.pattern_dropdown.value,
                min_confidence=0.5  # Get all signals, filter in backtest
            )
            
            if signals_result.signals_df.empty:
                with self.backtest_output:
                    clear_output(wait=True)
                    print(f"❌ No signals found for pattern: {self.pattern_dropdown.value}")
                return
            
            # Run backtest
            backtest_start = datetime.now()
            results = quick_backtest(
                strategy_config=strategy,
                signals_df=signals_result.signals_df,
                initial_capital=self.capital_input.value
            )
            backtest_time = (datetime.now() - backtest_start).total_seconds()
            
            self.backtest_results = results
            
            # Display results
            self._display_backtest_results(results, backtest_time)
            
        except Exception as e:
            with self.backtest_output:
                clear_output(wait=True)
                print(f"❌ Backtest failed: {e}")
            
            self.backtest_info.value = f"<div style='color: red;'>Backtest failed: {e}</div>"
    
    def _display_backtest_results(self, results: 'BacktestResults', execution_time: float):
        """Display comprehensive backtest results."""
        with self.backtest_output:
            clear_output(wait=True)
            
            print("📊 BACKTEST RESULTS")
            print("=" * 50)
            print(f"Strategy: {results.strategy_config.name}")
            print(f"Pattern: {results.strategy_config.pattern_id}")
            print(f"Period: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}")
            print(f"Execution Time: {execution_time:.2f} seconds")
            print()
            
            # Performance Metrics
            print("📈 PERFORMANCE METRICS")
            print("-" * 30)
            print(f"Initial Capital:      ${results.initial_capital:,.2f}")
            print(f"Final Capital:        ${results.final_capital:,.2f}")
            print(f"Total Return:         {results.total_return:.2f}%")
            print(f"CAGR:                 {results.cagr:.2f}%")
            print(f"Volatility:           {results.volatility:.2f}%")
            print(f"Sharpe Ratio:         {results.sharpe_ratio:.2f}")
            print(f"Max Drawdown:         {results.max_drawdown:.2f}%")
            print(f"Max DD Duration:      {results.max_drawdown_duration} days")
            print()
            
            # Trading Statistics
            print("📊 TRADING STATISTICS")
            print("-" * 30)
            print(f"Total Trades:         {results.total_trades}")
            print(f"Winning Trades:       {results.winning_trades}")
            print(f"Losing Trades:        {results.losing_trades}")
            print(f"Win Rate:             {results.win_rate:.1f}%")
            print(f"Average Win:          ${results.avg_win:.2f}")
            print(f"Average Loss:         ${results.avg_loss:.2f}")
            print(f"Profit Factor:        {results.profit_factor:.2f}")
            print(f"Avg Holding Days:     {results.avg_holding_days:.1f}")
            print()
            
            # Signal Processing
            print("🔍 SIGNAL PROCESSING")
            print("-" * 30)
            print(f"Signals Processed:    {results.signals_processed}")
            print(f"Signals Filtered:     {results.signals_filtered}")
            print(f"Signal Utilization:   {(results.signals_filtered/results.signals_processed*100) if results.signals_processed > 0 else 0:.1f}%")
            
            # Performance visualization
            if not results.equity_curve.empty:
                print("\n📈 Performance Chart:")
                
                plt.figure(figsize=(12, 8))
                
                # Equity curve
                plt.subplot(2, 1, 1)
                plt.plot(results.equity_curve.index, results.equity_curve['equity'], 
                        linewidth=2, color='steelblue', label='Portfolio Value')
                plt.axhline(y=results.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
                plt.title(f'Equity Curve - {results.strategy_config.name}')
                plt.ylabel('Portfolio Value ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Drawdown
                plt.subplot(2, 1, 2)
                plt.fill_between(results.drawdown_series.index, 
                               results.drawdown_series['drawdown_pct'], 0,
                               color='red', alpha=0.3, label='Drawdown')
                plt.title('Drawdown')
                plt.ylabel('Drawdown (%)')
                plt.xlabel('Date')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # Update info
        self.backtest_info.value = f"""
        <div style='background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;'>
            <b>Backtest Complete!</b> {results.total_trades} trades executed<br>
            <b>Performance:</b> {results.total_return:.1f}% return, {results.win_rate:.1f}% win rate<br>
            <b>Risk:</b> {results.max_drawdown:.1f}% max drawdown, {results.sharpe_ratio:.2f} Sharpe ratio
        </div>
        """
    
    def display_interface(self):
        """Display the backtesting interface."""
        
        # Header
        header = widgets.HTML(
            value="""
            <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='margin: 0; font-size: 24px;'>🚀 Quick Strategy Backtester</h2>
                <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>
                    Test pattern-based trading strategies with historical data
                </p>
            </div>
            """
        )
        
        # Controls
        controls_box = widgets.VBox([
            widgets.HTML("<h3>⚙️ Strategy Configuration</h3>"),
            widgets.HBox([
                self.strategy_dropdown,
                self.pattern_dropdown,
                self.capital_input
            ]),
            widgets.HBox([
                self.confidence_input,
                self.backtest_button
            ]),
            self.backtest_info
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            padding='15px',
            border_radius='5px',
            margin='10px 0px'
        ))
        
        # Results area
        results_box = widgets.VBox([
            widgets.HTML("<h3>📊 Backtest Results</h3>"),
            self.backtest_output
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            padding='15px',
            border_radius='5px',
            margin='10px 0px'
        ))
        
        # Display interface
        display(widgets.VBox([
            header,
            controls_box,
            results_box
        ]))

# %%
# Initialize and Display Quick Backtester
if signal_store:
    backtester = QuickBacktester(signal_store)
    backtester.display_interface()
else:
    display(widgets.HTML("""
    <div style='background: #ffe6e6; border: 1px solid #ff9999; padding: 20px; border-radius: 5px;'>
        <h3 style='color: #cc0000; margin-top: 0;'>⚠️ Backtester Not Available</h3>
        <p>The quick backtester requires pattern signal data to function properly.</p>
        <p>Please run pattern detection first to generate the required data.</p>
    </div>
    """))

# %%
# Utility Functions and Data Export
def export_pattern_examples(pattern_id: str, 
                          min_confidence: float = 0.75,
                          max_examples: int = 50,
                          output_format: str = 'csv') -> str:
    """
    Export pattern examples for external analysis.
    
    Args:
        pattern_id: Pattern identifier to export
        min_confidence: Minimum confidence threshold
        max_examples: Maximum number of examples
        output_format: Export format ('csv', 'json', 'excel')
        
    Returns:
        str: Path to exported file
    """
    try:
        # Query signals
        result = signal_store.read_signals(
            pattern_id=pattern_id,
            min_confidence=min_confidence,
            limit=max_examples
        )
        
        if result.signals_df.empty:
            print(f"No examples found for pattern {pattern_id}")
            return ""
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pattern_examples_{pattern_id}_{timestamp}"
        
        # Export based on format
        if output_format == 'csv':
            filepath = f"{filename}.csv"
            result.signals_df.to_csv(filepath, index=False)
        elif output_format == 'json':
            filepath = f"{filename}.json"
            result.signals_df.to_json(filepath, orient='records', date_format='iso')
        elif output_format == 'excel':
            filepath = f"{filename}.xlsx"
            result.signals_df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        
        print(f"✅ Exported {len(result.signals_df)} examples to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return ""

def get_pattern_statistics(days_lookback: int = 30) -> pd.DataFrame:
    """
    Get comprehensive pattern statistics for recent period.
    
    Args:
        days_lookback: Number of days to analyze
        
    Returns:
        pd.DataFrame: Pattern statistics summary
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_lookback)
        
        # Get all recent signals
        result = signal_store.read_signals(start_date=start_date, end_date=end_date)
        
        if result.signals_df.empty:
            print("No pattern data available for statistics")
            return pd.DataFrame()
        
        # Calculate statistics by pattern
        stats_list = []
        
        for pattern_id in result.patterns_found:
            pattern_signals = result.signals_df[result.signals_df['pattern_id'] == pattern_id]
            
            stats = {
                'pattern_id': pattern_id,
                'total_signals': len(pattern_signals),
                'avg_confidence': pattern_signals['confidence'].mean(),
                'std_confidence': pattern_signals['confidence'].std(),
                'min_confidence': pattern_signals['confidence'].min(),
                'max_confidence': pattern_signals['confidence'].max(),
                'unique_tickers': pattern_signals['ticker'].nunique(),
                'signals_per_day': len(pattern_signals) / days_lookback
            }
            
            stats_list.append(stats)
        
        stats_df = pd.DataFrame(stats_list)
        stats_df = stats_df.round(3)
        
        print(f"📊 Pattern Statistics ({days_lookback} days)")
        print("=" * 60)
        display(stats_df)
        
        return stats_df
        
    except Exception as e:
        print(f"❌ Statistics calculation failed: {e}")
        return pd.DataFrame()

# Display utility functions info
if signal_store:
    display(widgets.HTML("""
    <div style='background: #f0f8ff; border: 1px solid #b3d9ff; padding: 15px; border-radius: 5px; margin: 20px 0;'>
        <h3 style='margin-top: 0; color: #0066cc;'>🔧 Utility Functions Available</h3>
        <p><strong>Data Export:</strong></p>
        <code>export_pattern_examples('bear_flag', min_confidence=0.8, output_format='csv')</code>
        
        <p><strong>Pattern Statistics:</strong></p>
        <code>get_pattern_statistics(days_lookback=30)</code>
        
        <p>These functions can be used in other notebooks or scripts for further analysis.</p>
    </div>
    """))

print("✅ Pattern Explorer notebook loaded successfully!")
print("📊 Use the interactive widgets above to explore historical pattern examples")
print("🚀 Configure and run quick backtests to evaluate trading strategies")

# %% 