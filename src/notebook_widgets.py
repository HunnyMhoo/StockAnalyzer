"""
Notebook Widget Utilities Module

This module provides reusable UI components for Jupyter notebook interfaces,
extracted from the interactive demo for better maintainability and consistency
across notebooks.

Key Components:
- PatternAnalysisUI: Main UI creation class
- Widget factory functions for common patterns
- Event handler utilities
- Status management components
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from typing import Callable, Optional, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class WidgetConfig:
    """Configuration for widget styling and behavior"""
    description_width: str = 'initial'
    widget_width: str = '300px'
    button_width: str = '250px'
    button_height: str = '40px'
    textarea_height: str = '60px'
    status_padding: str = '10px'


class PatternAnalysisUI:
    """
    UI component factory for pattern analysis interfaces.
    
    Provides methods to create consistent, reusable UI components
    for pattern analysis notebooks.
    """
    
    def __init__(self, config: Optional[WidgetConfig] = None):
        """
        Initialize the UI factory.
        
        Args:
            config: Optional widget configuration
        """
        self.config = config or WidgetConfig()
        self.widgets = {}
        self.event_handlers = {}
        
    def create_input_widgets(self, defaults: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create input widgets for pattern analysis.
        
        Args:
            defaults: Default values for widgets
            
        Returns:
            Dict of created widgets
        """
        defaults = defaults or {
            'positive_ticker': '0700.HK',
            'start_date': '2024-01-15',
            'end_date': '2024-02-05',
            'negative_tickers': '0005.HK, 0941.HK, 0388.HK'
        }
        
        input_widgets = {
            'positive_ticker': widgets.Text(
                value=defaults['positive_ticker'],
                description='Positive Stock:',
                placeholder='e.g., 0700.HK',
                style={'description_width': self.config.description_width},
                layout=widgets.Layout(width=self.config.widget_width)
            ),
            
            'start_date': widgets.Text(
                value=defaults['start_date'],
                description='Pattern Start:',
                placeholder='YYYY-MM-DD',
                style={'description_width': self.config.description_width},
                layout=widgets.Layout(width=self.config.widget_width)
            ),
            
            'end_date': widgets.Text(
                value=defaults['end_date'],
                description='Pattern End:',
                placeholder='YYYY-MM-DD',
                style={'description_width': self.config.description_width},
                layout=widgets.Layout(width=self.config.widget_width)
            ),
            
            'negative_tickers': widgets.Textarea(
                value=defaults['negative_tickers'],
                description='Negative Examples:',
                placeholder='Comma-separated tickers (e.g., 0005.HK, 0001.HK)',
                style={'description_width': self.config.description_width},
                layout=widgets.Layout(width='400px', height=self.config.textarea_height)
            )
        }
        
        self.widgets.update(input_widgets)
        return input_widgets
    
    def create_configuration_widgets(self, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, widgets.Widget]:
        """
        Create configuration widgets for advanced options.
        
        Args:
            defaults: Default values for configuration
            
        Returns:
            Dict of created configuration widgets
        """
        defaults = defaults or {
            'min_confidence': 0.7,
            'max_stocks': 100
        }
        
        config_widgets = {
            'confidence_slider': widgets.FloatSlider(
                value=defaults['min_confidence'],
                min=0.5,
                max=0.95,
                step=0.05,
                description='Min Confidence:',
                style={'description_width': self.config.description_width},
                readout_format='.0%'
            ),
            
            'max_stocks_input': widgets.IntText(
                value=defaults['max_stocks'],
                description='Max Stocks to Scan:',
                style={'description_width': self.config.description_width},
                layout=widgets.Layout(width='200px')
            )
        }
        
        self.widgets.update(config_widgets)
        return config_widgets
    
    def create_control_widgets(self, button_text: str = "🔍 Find Similar Patterns") -> Dict[str, widgets.Widget]:
        """
        Create control widgets (buttons, status indicators).
        
        Args:
            button_text: Text for the main action button
            
        Returns:
            Dict of created control widgets
        """
        control_widgets = {
            'run_button': widgets.Button(
                description=button_text,
                button_style='primary',
                layout=widgets.Layout(
                    width=self.config.button_width, 
                    height=self.config.button_height
                ),
                tooltip='Start pattern scanning with current settings'
            ),
            
            'output_area': widgets.Output(),
            
            'status_html': widgets.HTML(
                value=f"<div style='padding: {self.config.status_padding}; background-color: #f0f0f0; border-radius: 5px;'>"
                      "<b>📊 Status:</b> Ready to scan. Configure your pattern above and click 'Find Similar Patterns'.</div>"
            )
        }
        
        self.widgets.update(control_widgets)
        return control_widgets
    
    def create_complete_interface(self, 
                                analyzer_function: Callable,
                                defaults: Optional[Dict[str, Any]] = None) -> widgets.VBox:
        """
        Create a complete pattern analysis interface.
        
        Args:
            analyzer_function: Function to call for pattern analysis
            defaults: Default values for all widgets
            
        Returns:
            Complete interface widget
        """
        # Set up defaults
        input_defaults = defaults.get('inputs', {}) if defaults else {}
        config_defaults = defaults.get('config', {}) if defaults else {}
        
        # Create widget groups
        input_widgets = self.create_input_widgets(input_defaults)
        config_widgets = self.create_configuration_widgets(config_defaults)
        control_widgets = self.create_control_widgets()
        
        # Set up event handler
        def on_button_click(b):
            """Enhanced button click handler with validation"""
            with control_widgets['output_area']:  # type: ignore
                clear_output(True)
                
                # Update status
                control_widgets['status_html'].value = (  # type: ignore
                    f"<div style='padding: {self.config.status_padding}; background-color: #fff3cd; border-radius: 5px;'>"
                    "<b>🔄 Status:</b> Scanning in progress... Please wait.</div>"
                )
                
                try:
                    # Simple validation
                    if not input_widgets['positive_ticker'].value.strip():  # type: ignore
                        raise ValueError("Please enter a positive ticker symbol.")
                    if not input_widgets['start_date'].value.strip() or not input_widgets['end_date'].value.strip():  # type: ignore
                        raise ValueError("Please enter both start and end dates.")
                    if not input_widgets['negative_tickers'].value.strip():  # type: ignore
                        raise ValueError("Please enter at least one negative example.")
                    
                    # Status update
                    print("🔍 **ENHANCED PATTERN SCANNING** (Refactored)")
                    print("=" * 55)
                    print(f"✅ Using new InteractivePatternAnalyzer module")
                    print(f"✅ Suppressing verbose logs for cleaner output")
                    print(f"✅ Progress tracking enabled")
                    print(f"✅ Confidence threshold: {config_widgets['confidence_slider'].value:.0%}")  # type: ignore
                    print(f"✅ Max stocks limit: {config_widgets['max_stocks_input'].value}")  # type: ignore
                    print()
                    
                    # Call the analyzer function
                    result = analyzer_function(
                        positive_ticker=input_widgets['positive_ticker'].value.strip(),  # type: ignore
                        start_date_str=input_widgets['start_date'].value.strip(),  # type: ignore
                        end_date_str=input_widgets['end_date'].value.strip(),  # type: ignore
                        negative_tickers_str=input_widgets['negative_tickers'].value.strip(),  # type: ignore
                        min_confidence=config_widgets['confidence_slider'].value,  # type: ignore
                        max_stocks_to_scan=config_widgets['max_stocks_input'].value  # type: ignore
                    )
                    
                    # Show additional metadata if available
                    if hasattr(result, 'success') and result.success:
                        print(f"\n📊 **Analysis Metadata:**")
                        print(f"   • Training samples: {result.analysis_metadata.get('training_samples', 'N/A')}")
                        print(f"   • Feature count: {result.analysis_metadata.get('feature_count', 'N/A')}")
                        print(f"   • Analysis time: {result.analysis_metadata.get('analysis_time', 0):.2f}s")
                    
                    # Update success status
                    control_widgets['status_html'].value = (  # type: ignore
                        f"<div style='padding: {self.config.status_padding}; background-color: #d1edff; border-radius: 5px;'>"
                        "<b>✅ Status:</b> Pattern scanning completed successfully!</div>"
                    )
                    
                except Exception as e:
                    print(f"❌ **Input Error:** {str(e)}")
                    control_widgets['status_html'].value = (  # type: ignore
                        f"<div style='padding: {self.config.status_padding}; background-color: #f8d7da; border-radius: 5px;'>"
                        f"<b>❌ Status:</b> Error - {str(e)}</div>"
                    )
        
        # Connect event handler
        control_widgets['run_button'].on_click(on_button_click)  # type: ignore
        
        # Assemble interface
        interface = widgets.VBox([
            widgets.HTML("<h3>🎯 Enhanced Pattern Definition</h3>"),
            widgets.HTML("<p><b>Define one positive example of the pattern you want to find:</b></p>"),
            
            widgets.HBox([
                input_widgets['positive_ticker'], 
                input_widgets['start_date'], 
                input_widgets['end_date']
            ]),
            
            widgets.HTML("<br><p><b>Provide negative examples (stocks that DON'T show this pattern):</b></p>"),
            input_widgets['negative_tickers'],
            
            widgets.HTML("<br><h3>⚙️ Enhanced Configuration</h3>"),
            widgets.HBox([
                config_widgets['confidence_slider'], 
                config_widgets['max_stocks_input']
            ]),
            
            widgets.HTML("<br>"),
            control_widgets['run_button'],
            control_widgets['status_html'],
            
            widgets.HTML("<br><h3>📊 Results</h3>"),
            control_widgets['output_area']
        ])
        
        return interface
    
    def get_widget_values(self) -> Dict[str, Any]:
        """
        Get current values from all widgets.
        
        Returns:
            Dict of widget values
        """
        values = {}
        for name, widget in self.widgets.items():
            if hasattr(widget, 'value'):
                values[name] = widget.value
        return values
    
    def update_status(self, message: str, status_type: str = 'info') -> None:
        """
        Update status message.
        
        Args:
            message: Status message
            status_type: Type of status ('info', 'success', 'warning', 'error')
        """
        if 'status_html' not in self.widgets:
            return
            
        colors = {
            'info': '#f0f0f0',
            'success': '#d1edff',
            'warning': '#fff3cd',
            'error': '#f8d7da'
        }
        
        color = colors.get(status_type, colors['info'])
        
        self.widgets['status_html'].value = (
            f"<div style='padding: {self.config.status_padding}; background-color: {color}; border-radius: 5px;'>"
            f"<b>📊 Status:</b> {message}</div>"
        )


def create_pattern_analysis_interface(analyzer_function: Callable, 
                                    defaults: Optional[Dict[str, Any]] = None) -> widgets.Widget:
    """
    Convenience function to create a complete pattern analysis interface.
    
    Args:
        analyzer_function: Function to call for pattern analysis
        defaults: Default values for widgets
        
    Returns:
        Complete interface widget
    """
    ui = PatternAnalysisUI()
    return ui.create_complete_interface(analyzer_function, defaults)


def create_simple_input_form(fields: Dict[str, Dict[str, Any]]) -> Dict[str, widgets.Widget]:
    """
    Create a simple input form with specified fields.
    
    Args:
        fields: Dict of field configurations
        
    Returns:
        Dict of created widgets
    """
    widgets_dict = {}
    
    for field_name, field_config in fields.items():
        widget_type = field_config.get('type', 'text')
        
        if widget_type == 'text':
            widget = widgets.Text(
                value=field_config.get('default', ''),
                description=field_config.get('label', field_name),
                placeholder=field_config.get('placeholder', ''),
                style={'description_width': 'initial'}
            )
        elif widget_type == 'textarea':
            widget = widgets.Textarea(
                value=field_config.get('default', ''),
                description=field_config.get('label', field_name),
                placeholder=field_config.get('placeholder', ''),
                style={'description_width': 'initial'}
            )
        elif widget_type == 'slider':
            widget = widgets.FloatSlider(
                value=field_config.get('default', 0.5),
                min=field_config.get('min', 0.0),
                max=field_config.get('max', 1.0),
                step=field_config.get('step', 0.1),
                description=field_config.get('label', field_name),
                style={'description_width': 'initial'}
            )
        elif widget_type == 'int':
            widget = widgets.IntText(
                value=field_config.get('default', 0),
                description=field_config.get('label', field_name),
                style={'description_width': 'initial'}
            )
        else:
            # Default to text widget
            widget = widgets.Text(
                value=field_config.get('default', ''),
                description=field_config.get('label', field_name),
                style={'description_width': 'initial'}
            )
        
        widgets_dict[field_name] = widget
    
    return widgets_dict 