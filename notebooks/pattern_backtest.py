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
# # Strategy Backtesting & Performance Analytics
# **Comprehensive Pattern-Based Strategy Evaluation**
#
# This notebook provides advanced backtesting capabilities for pattern-based trading strategies:
# - Strategy configuration and template management
# - Vectorized backtesting engine with risk management
# - Comprehensive performance analytics and metrics
# - Strategy comparison and optimization
# - Risk analysis and portfolio management
#
# **Target Users:** Portfolio Managers and Strategy Developers
# **Performance Goal:** Full backtest (5 years, HK universe) completes in < 90 seconds

# %%
# Import required libraries
import sys
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Interactive widgets
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Add project root to path
project_root = Path().resolve()
while not (project_root / 'stock_analyzer').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

try:
    # Stock analyzer imports
    from stock_analyzer.data.signal_store import SignalStore, query_signals
    from stock_analyzer.analysis.strategy import (
        StrategyManager, StrategyConfig, ExitRule, PositionSizing, 
        create_example_strategies
    )
    from stock_analyzer.analysis.backtester import (
        BacktestEngine, BacktestResults, quick_backtest, compare_strategies
    )
    from stock_analyzer.config import settings
    
    print("✅ Successfully imported stock_analyzer modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")

print(f"🚀 Strategy Backtester initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
# Configuration and Setup
class BacktestConfig:
    """Configuration for backtesting interface."""
    
    def __init__(self):
        # Interface settings
        self.chart_width = 1000
        self.chart_height = 600
        
        # Backtesting settings
        self.default_initial_capital = 100000.0
        self.default_commission = 0.1
        self.default_slippage = 0.05
        
        # Performance settings
        self.enable_parallel_processing = True
        self.chunk_size = 1000
        
        # Visualization settings
        self.color_scheme = {
            'equity': '#2E86C1',
            'drawdown': '#E74C3C',
            'benchmark': '#F39C12',
            'trades': '#28B463'
        }

# Initialize configuration and components
config = BacktestConfig()

# Initialize signal store and strategy manager
try:
    signal_store = SignalStore(base_dir="signals")
    strategy_manager = StrategyManager(strategies_dir="strategies")
    
    print(f"✅ Signal store initialized")
    print(f"✅ Strategy manager initialized")
    
    # Check available data
    patterns = signal_store.get_available_patterns()
    strategies = strategy_manager.list_strategies()
    
    if patterns:
        print(f"📈 Available patterns: {', '.join(patterns)}")
    if strategies:
        print(f"⚙️  Available strategies: {len(strategies)} saved strategies")
    
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    signal_store = None
    strategy_manager = None

# %%
# Advanced Strategy Builder Interface
class StrategyBuilder:
    """Interactive strategy configuration builder."""
    
    def __init__(self, strategy_manager: StrategyManager, available_patterns: List[str]):
        """Initialize strategy builder."""
        self.strategy_manager = strategy_manager
        self.available_patterns = available_patterns
        self.current_strategy = None
        
        self._create_widgets()
        self._setup_interactions()
    
    def _create_widgets(self):
        """Create strategy configuration widgets."""
        
        # Strategy basic info
        self.strategy_name = widgets.Text(
            value="Custom Strategy",
            description="Strategy Name:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.strategy_description = widgets.Textarea(
            value="Custom trading strategy configuration",
            description="Description:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px', height='80px')
        )
        
        # Pattern and entry conditions
        self.pattern_dropdown = widgets.Dropdown(
            options=self.available_patterns if self.available_patterns else ['bear_flag'],
            value=self.available_patterns[0] if self.available_patterns else 'bear_flag',
            description="Pattern:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        self.min_confidence = widgets.FloatSlider(
            value=0.75,
            min=0.5,
            max=0.95,
            step=0.05,
            description="Min Confidence:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'),
            readout_format='.2f'
        )
        
        self.max_signals_per_day = widgets.IntSlider(
            value=5,
            min=1,
            max=20,
            description="Max Signals/Day:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Position management
        self.holding_days = widgets.IntSlider(
            value=5,
            min=1,
            max=30,
            description="Holding Days:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.position_sizing = widgets.Dropdown(
            options=[
                ('Risk Percentage', 'risk_pct'),
                ('Fixed Amount', 'fixed_amount'),
                ('Volatility Target', 'volatility_target'),
                ('Kelly Criterion', 'kelly_criterion')
            ],
            value='risk_pct',
            description="Position Sizing:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        
        self.risk_pct = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=5.0,
            step=0.1,
            description="Risk % of Equity:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'),
            readout_format='.1f'
        )
        
        # Exit rules
        self.exit_rule = widgets.Dropdown(
            options=[
                ('Fixed Days', 'fixed_days'),
                ('Stop/Profit Targets', 'take_profit_stop_loss'),
                ('SMA Crossover', 'sma_cross'),
                ('Trailing Stop', 'trailing_stop')
            ],
            value='fixed_days',
            description="Exit Rule:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        
        self.stop_loss_pct = widgets.FloatSlider(
            value=2.0,
            min=0.5,
            max=10.0,
            step=0.5,
            description="Stop Loss %:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'),
            readout_format='.1f'
        )
        
        self.take_profit_pct = widgets.FloatSlider(
            value=4.0,
            min=1.0,
            max=20.0,
            step=0.5,
            description="Take Profit %:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'),
            readout_format='.1f'
        )
        
        self.sma_exit_period = widgets.IntSlider(
            value=20,
            min=5,
            max=100,
            description="SMA Exit Period:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.trailing_stop_pct = widgets.FloatSlider(
            value=3.0,
            min=1.0,
            max=10.0,
            step=0.5,
            description="Trailing Stop %:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'),
            readout_format='.1f'
        )
        
        # Risk management
        self.max_concurrent_positions = widgets.IntSlider(
            value=10,
            min=1,
            max=50,
            description="Max Positions:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.sector_concentration = widgets.FloatSlider(
            value=30.0,
            min=5.0,
            max=100.0,
            step=5.0,
            description="Max Sector %:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px'),
            readout_format='.0f'
        )
        
        # Strategy management buttons
        self.create_button = widgets.Button(
            description='Create Strategy',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        
        self.save_button = widgets.Button(
            description='Save Strategy',
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        
        self.load_button = widgets.Button(
            description='Load Strategy',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        
        # Template loader
        templates = self.strategy_manager.get_templates()
        self.template_dropdown = widgets.Dropdown(
            options=list(templates.keys()),
            description="Load Template:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        self.load_template_button = widgets.Button(
            description='Load Template',
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        
        # Strategy output
        self.strategy_output = widgets.Output()
        
        # Strategy info display
        self.strategy_info = widgets.HTML(
            value="<i>Configure parameters above and click 'Create Strategy'</i>"
        )
    
    def _setup_interactions(self):
        """Setup widget interactions."""
        self.create_button.on_click(self._on_create_strategy)
        self.save_button.on_click(self._on_save_strategy)
        self.load_template_button.on_click(self._on_load_template)
        
        # Dynamic widget visibility based on exit rule
        self.exit_rule.observe(self._on_exit_rule_changed, names='value')
        self._on_exit_rule_changed(None)  # Initialize visibility
    
    def _on_exit_rule_changed(self, change):
        """Handle exit rule changes to show/hide relevant widgets."""
        rule = self.exit_rule.value
        
        # Hide all exit-specific widgets first
        for widget in [self.stop_loss_pct, self.take_profit_pct, self.sma_exit_period, self.trailing_stop_pct]:
            widget.layout.display = 'none'
        
        # Show relevant widgets based on rule
        if rule == 'take_profit_stop_loss':
            self.stop_loss_pct.layout.display = 'block'
            self.take_profit_pct.layout.display = 'block'
        elif rule == 'sma_cross':
            self.sma_exit_period.layout.display = 'block'
        elif rule == 'trailing_stop':
            self.trailing_stop_pct.layout.display = 'block'
    
    def _on_create_strategy(self, button):
        """Handle create strategy button click."""
        try:
            # Build strategy configuration
            config_data = {
                'name': self.strategy_name.value,
                'description': self.strategy_description.value,
                'pattern_id': self.pattern_dropdown.value,
                'min_confidence': self.min_confidence.value,
                'max_signals_per_day': self.max_signals_per_day.value,
                'holding_days': self.holding_days.value,
                'position_sizing': self.position_sizing.value,
                'risk_pct_of_equity': self.risk_pct.value,
                'exit_rule': self.exit_rule.value,
                'max_concurrent_positions': self.max_concurrent_positions.value,
                'sector_concentration_limit': self.sector_concentration.value
            }
            
            # Add exit rule specific parameters
            if self.exit_rule.value == 'take_profit_stop_loss':
                config_data['stop_loss_pct'] = self.stop_loss_pct.value
                config_data['take_profit_pct'] = self.take_profit_pct.value
            elif self.exit_rule.value == 'sma_cross':
                config_data['sma_exit_period'] = self.sma_exit_period.value
            elif self.exit_rule.value == 'trailing_stop':
                config_data['trailing_stop_pct'] = self.trailing_stop_pct.value
            
            # Create strategy
            self.current_strategy = StrategyConfig(**config_data)
            
            with self.strategy_output:
                clear_output(wait=True)
                print("✅ Strategy created successfully!")
                print(f"Strategy: {self.current_strategy.name}")
                print(f"Pattern: {self.current_strategy.pattern_id}")
                print(f"Exit Rule: {self.current_strategy.exit_rule}")
                print(f"Risk Management: {self.current_strategy.risk_pct_of_equity}% per trade")
            
            self.strategy_info.value = f"""
            <div style='background: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px;'>
                <b>✅ Strategy Created:</b> {self.current_strategy.name}<br>
                <b>Pattern:</b> {self.current_strategy.pattern_id}<br>
                <b>Confidence:</b> {self.current_strategy.min_confidence:.2f}<br>
                <b>Exit Rule:</b> {self.current_strategy.exit_rule}
            </div>
            """
            
        except Exception as e:
            with self.strategy_output:
                clear_output(wait=True)
                print(f"❌ Strategy creation failed: {e}")
            
            self.strategy_info.value = f"<div style='color: red;'>Strategy creation failed: {e}</div>"
    
    def _on_save_strategy(self, button):
        """Handle save strategy button click."""
        if self.current_strategy is None:
            self.strategy_info.value = "<div style='color: red;'>No strategy to save. Create a strategy first.</div>"
            return
        
        try:
            file_path = self.strategy_manager.save_strategy(self.current_strategy)
            
            with self.strategy_output:
                clear_output(wait=True)
                print(f"✅ Strategy saved to: {file_path}")
            
            self.strategy_info.value = f"""
            <div style='background: #d1ecf1; border: 1px solid #bee5eb; padding: 10px; border-radius: 5px;'>
                <b>💾 Strategy Saved:</b> {self.current_strategy.name}<br>
                <b>File:</b> {Path(file_path).name}
            </div>
            """
            
        except Exception as e:
            with self.strategy_output:
                clear_output(wait=True)
                print(f"❌ Save failed: {e}")
            
            self.strategy_info.value = f"<div style='color: red;'>Save failed: {e}</div>"
    
    def _on_load_template(self, button):
        """Handle load template button click."""
        try:
            template_name = self.template_dropdown.value
            
            # Create strategy from template
            template_strategy = self.strategy_manager.create_strategy(
                template_name=template_name,
                pattern_id=self.pattern_dropdown.value
            )
            
            # Update widgets with template values
            self.strategy_name.value = template_strategy.name
            self.strategy_description.value = template_strategy.description
            self.min_confidence.value = template_strategy.min_confidence
            self.max_signals_per_day.value = template_strategy.max_signals_per_day
            self.holding_days.value = template_strategy.holding_days
            self.position_sizing.value = template_strategy.position_sizing
            self.risk_pct.value = template_strategy.risk_pct_of_equity
            self.exit_rule.value = template_strategy.exit_rule
            
            if template_strategy.stop_loss_pct:
                self.stop_loss_pct.value = template_strategy.stop_loss_pct
            if template_strategy.take_profit_pct:
                self.take_profit_pct.value = template_strategy.take_profit_pct
            if template_strategy.sma_exit_period:
                self.sma_exit_period.value = template_strategy.sma_exit_period
            if template_strategy.trailing_stop_pct:
                self.trailing_stop_pct.value = template_strategy.trailing_stop_pct
            
            self.max_concurrent_positions.value = template_strategy.max_concurrent_positions
            self.sector_concentration.value = template_strategy.sector_concentration_limit
            
            # Update exit rule visibility
            self._on_exit_rule_changed(None)
            
            with self.strategy_output:
                clear_output(wait=True)
                print(f"✅ Template '{template_name}' loaded successfully!")
                print("Modify parameters as needed and click 'Create Strategy'")
            
        except Exception as e:
            with self.strategy_output:
                clear_output(wait=True)
                print(f"❌ Template load failed: {e}")
    
    def display_interface(self):
        """Display the strategy builder interface."""
        
        # Header
        header = widgets.HTML(
            value="""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='margin: 0; font-size: 24px;'>⚙️ Strategy Builder</h2>
                <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>
                    Configure custom trading strategies with advanced parameters
                </p>
            </div>
            """
        )
        
        # Basic info section
        basic_info = widgets.VBox([
            widgets.HTML("<h3>📝 Basic Information</h3>"),
            widgets.HBox([self.strategy_name, self.pattern_dropdown]),
            self.strategy_description
        ])
        
        # Entry conditions section
        entry_conditions = widgets.VBox([
            widgets.HTML("<h3>🎯 Entry Conditions</h3>"),
            widgets.HBox([self.min_confidence, self.max_signals_per_day])
        ])
        
        # Position management section
        position_mgmt = widgets.VBox([
            widgets.HTML("<h3>💰 Position Management</h3>"),
            widgets.HBox([self.holding_days, self.position_sizing]),
            self.risk_pct
        ])
        
        # Exit rules section
        exit_rules = widgets.VBox([
            widgets.HTML("<h3>🚪 Exit Rules</h3>"),
            self.exit_rule,
            widgets.HBox([self.stop_loss_pct, self.take_profit_pct]),
            self.sma_exit_period,
            self.trailing_stop_pct
        ])
        
        # Risk management section
        risk_mgmt = widgets.VBox([
            widgets.HTML("<h3>🛡️ Risk Management</h3>"),
            widgets.HBox([self.max_concurrent_positions, self.sector_concentration])
        ])
        
        # Template and action buttons
        actions = widgets.VBox([
            widgets.HTML("<h3>🔧 Actions</h3>"),
            widgets.HBox([self.template_dropdown, self.load_template_button]),
            widgets.HBox([self.create_button, self.save_button, self.load_button]),
            self.strategy_info,
            self.strategy_output
        ])
        
        # Layout sections in tabs or accordion
        strategy_tabs = widgets.Tab()
        strategy_tabs.children = [
            widgets.VBox([basic_info, entry_conditions]),
            widgets.VBox([position_mgmt, exit_rules]),
            widgets.VBox([risk_mgmt, actions])
        ]
        strategy_tabs.set_title(0, 'Strategy Setup')
        strategy_tabs.set_title(1, 'Trading Rules')
        strategy_tabs.set_title(2, 'Risk & Actions')
        
        # Display complete interface
        display(widgets.VBox([header, strategy_tabs]))
        
        return self.current_strategy

# %%
# Initialize and Display Strategy Builder
if strategy_manager and signal_store:
    patterns = signal_store.get_available_patterns()
    builder = StrategyBuilder(strategy_manager, patterns)
    created_strategy = builder.display_interface()
else:
    display(widgets.HTML("""
    <div style='background: #ffe6e6; border: 1px solid #ff9999; padding: 20px; border-radius: 5px;'>
        <h3 style='color: #cc0000; margin-top: 0;'>⚠️ Strategy Builder Not Available</h3>
        <p>Strategy builder requires both signal store and strategy manager to be initialized.</p>
    </div>
    """))

# %% 