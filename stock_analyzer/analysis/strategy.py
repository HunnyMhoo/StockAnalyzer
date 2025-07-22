"""
Strategy Configuration Management Module

This module provides strategy configuration, validation, and persistence functionality
for pattern-based trading strategies using YAML format.

Key Features:
- Strategy parameter validation and type checking
- YAML serialization/deserialization
- Strategy template management
- Version tracking and metadata
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum

import yaml
from pydantic import BaseModel, Field, field_validator

# Handle imports for both direct execution and package usage
try:
    from ..config import settings
except ImportError:
    from stock_analyzer.config import settings


class StrategyError(Exception):
    """Custom exception for strategy operations."""
    pass


class ExitRule(str, Enum):
    """Available exit rules for strategies."""
    FIXED_DAYS = "fixed_days"
    SMA_CROSS = "sma_cross"
    TAKE_PROFIT_STOP_LOSS = "take_profit_stop_loss"
    TRAILING_STOP = "trailing_stop"


class PositionSizing(str, Enum):
    """Position sizing methods."""
    FIXED_AMOUNT = "fixed_amount"
    RISK_PCT = "risk_pct"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGET = "volatility_target"


class StrategyConfig(BaseModel):
    """
    Pydantic model for strategy configuration validation.
    """
    # Strategy metadata
    name: str = Field(description="Strategy name")
    description: str = Field(default="", description="Strategy description")
    pattern_id: str = Field(description="Pattern identifier to trade")
    version: str = Field(default="1.0", description="Strategy version")
    created_date: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    modified_date: datetime = Field(default_factory=datetime.now, description="Last modification timestamp")
    
    # Entry conditions
    min_confidence: float = Field(ge=0.0, le=1.0, description="Minimum pattern confidence threshold")
    max_signals_per_day: int = Field(default=5, ge=1, description="Maximum signals to act on per day")
    
    # Position management
    holding_days: int = Field(default=5, ge=1, description="Default holding period in days")
    position_sizing: PositionSizing = Field(default=PositionSizing.RISK_PCT, description="Position sizing method")
    risk_pct_of_equity: float = Field(default=1.0, ge=0.1, le=10.0, description="Risk percentage of equity per trade")
    
    # Exit rules
    exit_rule: ExitRule = Field(default=ExitRule.FIXED_DAYS, description="Exit rule type")
    stop_loss_pct: Optional[float] = Field(default=None, ge=0.1, le=50.0, description="Stop loss percentage")
    take_profit_pct: Optional[float] = Field(default=None, ge=0.1, le=100.0, description="Take profit percentage")
    trailing_stop_pct: Optional[float] = Field(default=None, ge=0.1, le=50.0, description="Trailing stop percentage")
    sma_exit_period: Optional[int] = Field(default=None, ge=5, le=200, description="SMA period for exit signal")
    
    # Risk management
    max_concurrent_positions: int = Field(default=10, ge=1, description="Maximum concurrent positions")
    sector_concentration_limit: float = Field(default=30.0, ge=5.0, le=100.0, description="Max percentage in single sector")
    
    # Backtesting parameters
    start_date: Optional[datetime] = Field(default=None, description="Backtest start date")
    end_date: Optional[datetime] = Field(default=None, description="Backtest end date")
    initial_capital: float = Field(default=100000.0, ge=1000.0, description="Initial capital for backtesting")
    
    # Commission and slippage (for future use)
    commission_pct: float = Field(default=0.1, ge=0.0, le=1.0, description="Commission percentage")
    slippage_pct: float = Field(default=0.05, ge=0.0, le=1.0, description="Slippage percentage")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate strategy name format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Strategy name cannot be empty")
        # Check for valid filename characters
        invalid_chars = '<>:"/\\|?*'
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Strategy name contains invalid characters: {invalid_chars}")
        return v.strip()
    
    @field_validator('pattern_id')
    @classmethod
    def validate_pattern_id(cls, v):
        """Validate pattern ID format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Pattern ID cannot be empty")
        return v.strip().lower()
    
    @field_validator('stop_loss_pct', 'take_profit_pct')
    @classmethod
    def validate_stop_take_profit(cls, v, info):
        """Validate stop loss and take profit relationships."""
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v
    
    def model_post_init(self, __context) -> None:
        """Post-initialization validation."""
        # Update modified date
        self.modified_date = datetime.now()
        
        # Validate exit rule parameters
        if self.exit_rule == ExitRule.TAKE_PROFIT_STOP_LOSS:
            if self.stop_loss_pct is None and self.take_profit_pct is None:
                raise ValueError("Either stop_loss_pct or take_profit_pct must be set for take_profit_stop_loss exit rule")
        
        if self.exit_rule == ExitRule.SMA_CROSS:
            if self.sma_exit_period is None:
                raise ValueError("sma_exit_period must be set for sma_cross exit rule")
        
        if self.exit_rule == ExitRule.TRAILING_STOP:
            if self.trailing_stop_pct is None:
                raise ValueError("trailing_stop_pct must be set for trailing_stop exit rule")


@dataclass
class StrategyTemplate:
    """Template for creating common strategy configurations."""
    name: str
    description: str
    config_updates: Dict[str, Any]
    
    def create_config(self, pattern_id: str, **overrides) -> StrategyConfig:
        """Create a strategy config from this template."""
        # Start with default config
        config_data = {
            'name': self.name,
            'description': self.description,
            'pattern_id': pattern_id
        }
        
        # Apply template updates
        config_data.update(self.config_updates)
        
        # Apply any overrides
        config_data.update(overrides)
        
        return StrategyConfig(**config_data)


class StrategyManager:
    """
    Strategy configuration manager with YAML persistence.
    
    This class provides functionality to:
    - Create and validate strategy configurations
    - Save/load strategies to/from YAML files
    - Manage strategy templates
    - Track strategy versions and metadata
    """
    
    def __init__(self, strategies_dir: str = "strategies"):
        """
        Initialize StrategyManager.
        
        Args:
            strategies_dir: Directory for strategy files
        """
        self.strategies_dir = Path(strategies_dir)
        self.templates = self._load_default_templates()
        
        # Create directory if it doesn't exist
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_default_templates(self) -> Dict[str, StrategyTemplate]:
        """Load default strategy templates."""
        templates = {
            'conservative': StrategyTemplate(
                name='Conservative Pattern Trading',
                description='Low-risk strategy with tight stops and high confidence threshold',
                config_updates={
                    'min_confidence': 0.85,
                    'holding_days': 3,
                    'risk_pct_of_equity': 0.5,
                    'stop_loss_pct': 2.0,
                    'take_profit_pct': 4.0,
                    'exit_rule': ExitRule.TAKE_PROFIT_STOP_LOSS,
                    'max_concurrent_positions': 5
                }
            ),
            'aggressive': StrategyTemplate(
                name='Aggressive Pattern Trading',
                description='Higher-risk strategy with wider stops and lower confidence threshold',
                config_updates={
                    'min_confidence': 0.70,
                    'holding_days': 7,
                    'risk_pct_of_equity': 2.0,
                    'stop_loss_pct': 5.0,
                    'take_profit_pct': 10.0,
                    'exit_rule': ExitRule.TAKE_PROFIT_STOP_LOSS,
                    'max_concurrent_positions': 15
                }
            ),
            'trend_following': StrategyTemplate(
                name='Trend Following Pattern Strategy',
                description='Hold positions longer with SMA-based exits',
                config_updates={
                    'min_confidence': 0.75,
                    'holding_days': 10,
                    'risk_pct_of_equity': 1.5,
                    'sma_exit_period': 20,
                    'exit_rule': ExitRule.SMA_CROSS,
                    'max_concurrent_positions': 8
                }
            ),
            'scalping': StrategyTemplate(
                name='Quick Scalping Strategy',
                description='Short-term trades with quick exits',
                config_updates={
                    'min_confidence': 0.80,
                    'holding_days': 1,
                    'risk_pct_of_equity': 0.8,
                    'take_profit_pct': 2.0,
                    'stop_loss_pct': 1.0,
                    'exit_rule': ExitRule.TAKE_PROFIT_STOP_LOSS,
                    'max_signals_per_day': 10
                }
            )
        }
        return templates
    
    def create_strategy(self, 
                       template_name: Optional[str] = None,
                       pattern_id: str = "generic_pattern",
                       **config_overrides) -> StrategyConfig:
        """
        Create a new strategy configuration.
        
        Args:
            template_name: Name of template to use (None for default)
            pattern_id: Pattern identifier for the strategy
            **config_overrides: Override default configuration values
            
        Returns:
            StrategyConfig: Validated strategy configuration
        """
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            return template.create_config(pattern_id, **config_overrides)
        else:
            # Create default strategy
            config_data = {
                'name': f'Pattern Strategy - {pattern_id}',
                'pattern_id': pattern_id,
                **config_overrides
            }
            return StrategyConfig(**config_data)
    
    def save_strategy(self, strategy: StrategyConfig, filename: Optional[str] = None) -> str:
        """
        Save strategy configuration to YAML file.
        
        Args:
            strategy: Strategy configuration to save
            filename: Custom filename (auto-generated if None)
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            # Generate filename from strategy name and pattern
            safe_name = "".join(c for c in strategy.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_').lower()
            filename = f"{safe_name}_{strategy.pattern_id}.yaml"
        
        file_path = self.strategies_dir / filename
        
        try:
            # Convert to dictionary for YAML serialization
            strategy_dict = strategy.model_dump()
            
            # Convert datetime objects to ISO format strings
            for key, value in strategy_dict.items():
                if isinstance(value, datetime):
                    strategy_dict[key] = value.isoformat()
            
            # Add metadata
            strategy_dict['_metadata'] = {
                'saved_by': 'StrategyManager',
                'saved_at': datetime.now().isoformat(),
                'version_info': strategy.version
            }
            
            # Save to YAML
            with open(file_path, 'w') as f:
                yaml.dump(strategy_dict, f, default_flow_style=False, sort_keys=False)
            
            print(f"✓ Strategy saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            raise StrategyError(f"Failed to save strategy: {e}")
    
    def load_strategy(self, filename: str) -> StrategyConfig:
        """
        Load strategy configuration from YAML file.
        
        Args:
            filename: Name of strategy file to load
            
        Returns:
            StrategyConfig: Loaded strategy configuration
        """
        file_path = self.strategies_dir / filename
        
        if not file_path.exists():
            raise StrategyError(f"Strategy file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                strategy_dict = yaml.safe_load(f)
            
            # Remove metadata before validation
            if '_metadata' in strategy_dict:
                del strategy_dict['_metadata']
            
            # Convert ISO date strings back to datetime objects and handle type conversions
            for key, value in strategy_dict.items():
                if isinstance(value, str) and key.endswith('_date'):
                    try:
                        strategy_dict[key] = datetime.fromisoformat(value)
                    except ValueError:
                        pass  # Keep as string if not valid ISO format
                elif key in ['min_confidence', 'risk_pct_of_equity', 'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct', 'sector_concentration_limit', 'initial_capital', 'commission_pct', 'slippage_pct']:
                    # Convert numeric fields from string if needed
                    if isinstance(value, str) and value.strip():
                        try:
                            strategy_dict[key] = float(value)
                        except ValueError:
                            pass
                elif key in ['max_signals_per_day', 'holding_days', 'max_concurrent_positions', 'sma_exit_period']:
                    # Convert integer fields from string if needed
                    if isinstance(value, str) and value.strip():
                        try:
                            strategy_dict[key] = int(value)
                        except ValueError:
                            pass
                elif key in ['position_sizing', 'exit_rule']:
                    # Enum fields should be strings, but validate they're proper enum values
                    if isinstance(value, str):
                        continue  # Let Pydantic handle enum validation
            
            # Create and validate strategy config
            strategy = StrategyConfig(**strategy_dict)
            
            print(f"✓ Strategy loaded from: {file_path}")
            return strategy
            
        except Exception as e:
            raise StrategyError(f"Failed to load strategy from {filename}: {e}")
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """
        List all available strategy files with metadata.
        
        Returns:
            List[Dict]: Strategy file information
        """
        strategies = []
        
        for file_path in self.strategies_dir.glob("*.yaml"):
            try:
                with open(file_path, 'r') as f:
                    strategy_dict = yaml.safe_load(f)
                
                strategy_info = {
                    'filename': file_path.name,
                    'name': strategy_dict.get('name', 'Unknown'),
                    'pattern_id': strategy_dict.get('pattern_id', 'Unknown'),
                    'version': strategy_dict.get('version', '1.0'),
                    'created_date': strategy_dict.get('created_date'),
                    'modified_date': strategy_dict.get('modified_date'),
                    'description': strategy_dict.get('description', '')
                }
                
                strategies.append(strategy_info)
                
            except Exception as e:
                warnings.warn(f"Failed to read strategy file {file_path.name}: {e}")
                continue
        
        return sorted(strategies, key=lambda x: x['modified_date'] or '', reverse=True)
    
    def get_templates(self) -> Dict[str, str]:
        """Get available strategy templates with descriptions."""
        return {name: template.description for name, template in self.templates.items()}
    
    def delete_strategy(self, filename: str) -> bool:
        """
        Delete a strategy file.
        
        Args:
            filename: Name of strategy file to delete
            
        Returns:
            bool: True if deleted successfully
        """
        file_path = self.strategies_dir / filename
        
        if not file_path.exists():
            warnings.warn(f"Strategy file not found: {filename}")
            return False
        
        try:
            file_path.unlink()
            print(f"✓ Deleted strategy: {filename}")
            return True
        except Exception as e:
            warnings.warn(f"Failed to delete strategy {filename}: {e}")
            return False
    
    def validate_strategy(self, strategy_dict: Dict[str, Any]) -> List[str]:
        """
        Validate strategy configuration without creating object.
        
        Args:
            strategy_dict: Strategy configuration dictionary
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Try to create StrategyConfig to validate
            StrategyConfig(**strategy_dict)
        except Exception as e:
            errors.append(str(e))
        
        return errors


def create_example_strategies(manager: StrategyManager, pattern_id: str = "bear_flag") -> List[str]:
    """
    Create example strategies for demonstration.
    
    Args:
        manager: StrategyManager instance
        pattern_id: Pattern ID to use for examples
        
    Returns:
        List[str]: Paths to created strategy files
    """
    created_files = []
    
    # Create one strategy from each template
    for template_name in manager.templates.keys():
        try:
            strategy = manager.create_strategy(
                template_name=template_name,
                pattern_id=pattern_id,
                name=f"{manager.templates[template_name].name} - {pattern_id.title()}"
            )
            
            file_path = manager.save_strategy(strategy)
            created_files.append(file_path)
            
        except Exception as e:
            warnings.warn(f"Failed to create example strategy {template_name}: {e}")
    
    return created_files


# Convenience functions
def save_strategy_config(strategy: StrategyConfig, 
                        filename: Optional[str] = None,
                        strategies_dir: str = "strategies") -> str:
    """Convenience function to save a strategy configuration."""
    manager = StrategyManager(strategies_dir)
    return manager.save_strategy(strategy, filename)


def load_strategy_config(filename: str, 
                        strategies_dir: str = "strategies") -> StrategyConfig:
    """Convenience function to load a strategy configuration."""
    manager = StrategyManager(strategies_dir)
    return manager.load_strategy(filename) 