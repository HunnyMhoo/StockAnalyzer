"""
Unit tests for the Strategy configuration module.

Tests cover strategy validation, templates, YAML persistence, and error handling.
"""

import os
import pytest
import tempfile
import shutil
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, mock_open

from stock_analyzer.analysis.strategy import (
    StrategyConfig,
    StrategyManager,
    StrategyTemplate,
    StrategyError,
    ExitRule,
    PositionSizing,
    create_example_strategies,
    save_strategy_config,
    load_strategy_config
)


class TestStrategyConfig:
    """Test StrategyConfig validation and functionality."""
    
    def test_valid_strategy_config(self):
        """Test creating a valid strategy configuration."""
        config = StrategyConfig(
            name="Test Strategy",
            pattern_id="bear_flag",
            min_confidence=0.85,
            holding_days=5,
            risk_pct_of_equity=1.5
        )
        
        assert config.name == "Test Strategy"
        assert config.pattern_id == "bear_flag"
        assert config.min_confidence == 0.85
        assert config.holding_days == 5
        assert config.risk_pct_of_equity == 1.5
        assert config.exit_rule == ExitRule.FIXED_DAYS  # Default
        assert config.position_sizing == PositionSizing.RISK_PCT  # Default
    
    def test_strategy_name_validation(self):
        """Test strategy name validation."""
        # Valid name
        config = StrategyConfig(
            name="  Valid Strategy Name  ",
            pattern_id="bear_flag",
            min_confidence=0.85
        )
        assert config.name == "Valid Strategy Name"
        
        # Empty name should raise error
        with pytest.raises(ValueError, match="Strategy name cannot be empty"):
            StrategyConfig(
                name="",
                pattern_id="bear_flag",
                min_confidence=0.85
            )
        
        # Name with invalid characters should raise error
        with pytest.raises(ValueError, match="Strategy name contains invalid characters"):
            StrategyConfig(
                name="Invalid<Name>",
                pattern_id="bear_flag",
                min_confidence=0.85
            )
    
    def test_pattern_id_validation(self):
        """Test pattern ID validation and normalization."""
        # Valid pattern ID should be normalized to lowercase
        config = StrategyConfig(
            name="Test Strategy",
            pattern_id="  BEAR_FLAG  ",
            min_confidence=0.85
        )
        assert config.pattern_id == "bear_flag"
        
        # Empty pattern ID should raise error
        with pytest.raises(ValueError, match="Pattern ID cannot be empty"):
            StrategyConfig(
                name="Test Strategy",
                pattern_id="",
                min_confidence=0.85
            )
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        for confidence in [0.0, 0.5, 1.0]:
            config = StrategyConfig(
                name="Test Strategy",
                pattern_id="bear_flag",
                min_confidence=confidence
            )
            assert config.min_confidence == confidence
        
        # Invalid confidence scores should raise error
        for invalid_confidence in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValueError):
                StrategyConfig(
                    name="Test Strategy",
                    pattern_id="bear_flag",
                    min_confidence=invalid_confidence
                )
    
    def test_risk_percentage_validation(self):
        """Test risk percentage validation."""
        # Valid risk percentages
        for risk_pct in [0.1, 1.0, 5.0, 10.0]:
            config = StrategyConfig(
                name="Test Strategy",
                pattern_id="bear_flag",
                min_confidence=0.85,
                risk_pct_of_equity=risk_pct
            )
            assert config.risk_pct_of_equity == risk_pct
        
        # Invalid risk percentages should raise error
        for invalid_risk in [0.05, 15.0]:
            with pytest.raises(ValueError):
                StrategyConfig(
                    name="Test Strategy",
                    pattern_id="bear_flag",
                    min_confidence=0.85,
                    risk_pct_of_equity=invalid_risk
                )
    
    def test_exit_rule_validation(self):
        """Test exit rule validation and dependencies."""
        # TAKE_PROFIT_STOP_LOSS rule with stop loss
        config = StrategyConfig(
            name="Test Strategy",
            pattern_id="bear_flag",
            min_confidence=0.85,
            exit_rule=ExitRule.TAKE_PROFIT_STOP_LOSS,
            stop_loss_pct=2.0
        )
        assert config.exit_rule == ExitRule.TAKE_PROFIT_STOP_LOSS
        assert config.stop_loss_pct == 2.0
        
        # TAKE_PROFIT_STOP_LOSS rule with take profit
        config = StrategyConfig(
            name="Test Strategy",
            pattern_id="bear_flag",
            min_confidence=0.85,
            exit_rule=ExitRule.TAKE_PROFIT_STOP_LOSS,
            take_profit_pct=4.0
        )
        assert config.take_profit_pct == 4.0
        
        # TAKE_PROFIT_STOP_LOSS rule without either should raise error
        with pytest.raises(ValueError, match="Either stop_loss_pct or take_profit_pct must be set"):
            StrategyConfig(
                name="Test Strategy",
                pattern_id="bear_flag",
                min_confidence=0.85,
                exit_rule=ExitRule.TAKE_PROFIT_STOP_LOSS
            )
        
        # SMA_CROSS rule with period
        config = StrategyConfig(
            name="Test Strategy",
            pattern_id="bear_flag",
            min_confidence=0.85,
            exit_rule=ExitRule.SMA_CROSS,
            sma_exit_period=20
        )
        assert config.sma_exit_period == 20
        
        # SMA_CROSS rule without period should raise error
        with pytest.raises(ValueError, match="sma_exit_period must be set for sma_cross exit rule"):
            StrategyConfig(
                name="Test Strategy",
                pattern_id="bear_flag",
                min_confidence=0.85,
                exit_rule=ExitRule.SMA_CROSS
            )
        
        # TRAILING_STOP rule with percentage
        config = StrategyConfig(
            name="Test Strategy",
            pattern_id="bear_flag",
            min_confidence=0.85,
            exit_rule=ExitRule.TRAILING_STOP,
            trailing_stop_pct=3.0
        )
        assert config.trailing_stop_pct == 3.0
        
        # TRAILING_STOP rule without percentage should raise error
        with pytest.raises(ValueError, match="trailing_stop_pct must be set for trailing_stop exit rule"):
            StrategyConfig(
                name="Test Strategy",
                pattern_id="bear_flag",
                min_confidence=0.85,
                exit_rule=ExitRule.TRAILING_STOP
            )
    
    def test_stop_take_profit_validation(self):
        """Test stop loss and take profit validation."""
        # Valid positive values
        config = StrategyConfig(
            name="Test Strategy",
            pattern_id="bear_flag",
            min_confidence=0.85,
            stop_loss_pct=2.0,
            take_profit_pct=4.0
        )
        assert config.stop_loss_pct == 2.0
        assert config.take_profit_pct == 4.0
        
        # Zero or negative values should raise error
        with pytest.raises(ValueError):
            StrategyConfig(
                name="Test Strategy",
                pattern_id="bear_flag",
                min_confidence=0.85,
                stop_loss_pct=0.0
            )
        
        with pytest.raises(ValueError):
            StrategyConfig(
                name="Test Strategy",
                pattern_id="bear_flag",
                min_confidence=0.85,
                take_profit_pct=-1.0
            )
    
    def test_modified_date_update(self):
        """Test that modified_date is updated on creation."""
        before_creation = datetime.now()
        
        config = StrategyConfig(
            name="Test Strategy",
            pattern_id="bear_flag",
            min_confidence=0.85
        )
        
        after_creation = datetime.now()
        
        assert before_creation <= config.modified_date <= after_creation


class TestStrategyTemplate:
    """Test StrategyTemplate functionality."""
    
    def test_create_config_from_template(self):
        """Test creating config from template."""
        template = StrategyTemplate(
            name="Conservative Template",
            description="Low-risk strategy",
            config_updates={
                'min_confidence': 0.90,
                'risk_pct_of_equity': 0.5,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'exit_rule': ExitRule.TAKE_PROFIT_STOP_LOSS
            }
        )
        
        config = template.create_config(pattern_id="bear_flag")
        
        assert config.name == "Conservative Template"
        assert config.description == "Low-risk strategy"
        assert config.pattern_id == "bear_flag"
        assert config.min_confidence == 0.90
        assert config.risk_pct_of_equity == 0.5
        assert config.stop_loss_pct == 2.0
        assert config.take_profit_pct == 4.0
        assert config.exit_rule == ExitRule.TAKE_PROFIT_STOP_LOSS
    
    def test_create_config_with_overrides(self):
        """Test creating config with parameter overrides."""
        template = StrategyTemplate(
            name="Template",
            description="Base template",
            config_updates={
                'min_confidence': 0.85,
                'risk_pct_of_equity': 1.0
            }
        )
        
        config = template.create_config(
            pattern_id="bear_flag",
            min_confidence=0.95,  # Override template value
            holding_days=10  # Add new parameter
        )
        
        assert config.min_confidence == 0.95  # Overridden
        assert config.risk_pct_of_equity == 1.0  # From template
        assert config.holding_days == 10  # Added


class TestStrategyManager:
    """Test StrategyManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def strategy_manager(self, temp_dir):
        """Create a StrategyManager instance for testing."""
        return StrategyManager(strategies_dir=temp_dir)
    
    @pytest.fixture
    def sample_strategy(self):
        """Create a sample strategy configuration."""
        return StrategyConfig(
            name="Sample Strategy",
            description="Test strategy for unit tests",
            pattern_id="bear_flag",
            min_confidence=0.85,
            holding_days=5,
            risk_pct_of_equity=1.5,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            exit_rule=ExitRule.TAKE_PROFIT_STOP_LOSS
        )
    
    def test_initialization(self, temp_dir):
        """Test StrategyManager initialization."""
        manager = StrategyManager(strategies_dir=temp_dir)
        
        assert manager.strategies_dir == Path(temp_dir)
        assert Path(temp_dir).exists()
        assert len(manager.templates) > 0
        assert 'conservative' in manager.templates
        assert 'aggressive' in manager.templates
    
    def test_create_strategy_default(self, strategy_manager):
        """Test creating a default strategy."""
        strategy = strategy_manager.create_strategy(pattern_id="bear_flag")
        
        assert strategy.name == "Pattern Strategy - bear_flag"
        assert strategy.pattern_id == "bear_flag"
        assert isinstance(strategy, StrategyConfig)
    
    def test_create_strategy_from_template(self, strategy_manager):
        """Test creating strategy from template."""
        strategy = strategy_manager.create_strategy(
            template_name="conservative",
            pattern_id="bear_flag"
        )
        
        assert strategy.pattern_id == "bear_flag"
        assert strategy.min_confidence == 0.85  # From conservative template
        assert strategy.risk_pct_of_equity == 0.5  # From conservative template
        assert strategy.exit_rule == ExitRule.TAKE_PROFIT_STOP_LOSS
    
    def test_create_strategy_with_overrides(self, strategy_manager):
        """Test creating strategy with parameter overrides."""
        strategy = strategy_manager.create_strategy(
            template_name="aggressive",
            pattern_id="bear_flag",
            name="Custom Aggressive Strategy",
            min_confidence=0.80
        )
        
        assert strategy.name == "Custom Aggressive Strategy"
        assert strategy.min_confidence == 0.80  # Overridden
        assert strategy.risk_pct_of_equity == 2.0  # From aggressive template
    
    def test_save_strategy(self, strategy_manager, sample_strategy):
        """Test saving strategy to YAML file."""
        file_path = strategy_manager.save_strategy(sample_strategy)
        
        assert os.path.exists(file_path)
        assert file_path.endswith('.yaml')
        
        # Verify YAML content
        with open(file_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        assert yaml_content['name'] == sample_strategy.name
        assert yaml_content['pattern_id'] == sample_strategy.pattern_id
        assert yaml_content['min_confidence'] == sample_strategy.min_confidence
        assert '_metadata' in yaml_content
    
    def test_save_strategy_custom_filename(self, strategy_manager, sample_strategy):
        """Test saving strategy with custom filename."""
        custom_filename = "my_custom_strategy.yaml"
        file_path = strategy_manager.save_strategy(sample_strategy, filename=custom_filename)
        
        assert file_path.endswith(custom_filename)
        assert os.path.exists(file_path)
    
    def test_load_strategy(self, strategy_manager, sample_strategy):
        """Test loading strategy from YAML file."""
        # Save strategy first
        file_path = strategy_manager.save_strategy(sample_strategy)
        filename = os.path.basename(file_path)
        
        # Load strategy back
        loaded_strategy = strategy_manager.load_strategy(filename)
        
        assert loaded_strategy.name == sample_strategy.name
        assert loaded_strategy.pattern_id == sample_strategy.pattern_id
        assert loaded_strategy.min_confidence == sample_strategy.min_confidence
        assert loaded_strategy.holding_days == sample_strategy.holding_days
        assert loaded_strategy.exit_rule == sample_strategy.exit_rule
    
    def test_load_nonexistent_strategy(self, strategy_manager):
        """Test loading a non-existent strategy file."""
        with pytest.raises(StrategyError, match="Strategy file not found"):
            strategy_manager.load_strategy("nonexistent.yaml")
    
    def test_list_strategies(self, strategy_manager, sample_strategy):
        """Test listing available strategies."""
        # Initially should be empty
        strategies = strategy_manager.list_strategies()
        assert len(strategies) == 0
        
        # Save a strategy and check list
        strategy_manager.save_strategy(sample_strategy)
        strategies = strategy_manager.list_strategies()
        
        assert len(strategies) == 1
        strategy_info = strategies[0]
        assert strategy_info['name'] == sample_strategy.name
        assert strategy_info['pattern_id'] == sample_strategy.pattern_id
        assert 'filename' in strategy_info
        assert 'version' in strategy_info
        assert 'created_date' in strategy_info
    
    def test_get_templates(self, strategy_manager):
        """Test getting available templates."""
        templates = strategy_manager.get_templates()
        
        assert isinstance(templates, dict)
        assert len(templates) > 0
        assert 'conservative' in templates
        assert 'aggressive' in templates
        assert 'trend_following' in templates
        assert 'scalping' in templates
        
        # Verify descriptions are strings
        for description in templates.values():
            assert isinstance(description, str)
            assert len(description) > 0
    
    def test_delete_strategy(self, strategy_manager, sample_strategy):
        """Test deleting a strategy file."""
        # Save strategy first
        file_path = strategy_manager.save_strategy(sample_strategy)
        filename = os.path.basename(file_path)
        
        # Verify file exists
        assert os.path.exists(file_path)
        
        # Delete strategy
        result = strategy_manager.delete_strategy(filename)
        
        assert result is True
        assert not os.path.exists(file_path)
    
    def test_delete_nonexistent_strategy(self, strategy_manager):
        """Test deleting a non-existent strategy file."""
        result = strategy_manager.delete_strategy("nonexistent.yaml")
        assert result is False
    
    def test_validate_strategy(self, strategy_manager):
        """Test strategy validation."""
        # Valid strategy dictionary
        valid_dict = {
            'name': 'Test Strategy',
            'pattern_id': 'bear_flag',
            'min_confidence': 0.85,
            'holding_days': 5,
            'risk_pct_of_equity': 1.5
        }
        
        errors = strategy_manager.validate_strategy(valid_dict)
        assert len(errors) == 0
        
        # Invalid strategy dictionary
        invalid_dict = {
            'name': 'Test Strategy',
            'pattern_id': 'bear_flag',
            'min_confidence': 1.5,  # Invalid confidence > 1.0
            'risk_pct_of_equity': 0.05  # Invalid risk < 0.1
        }
        
        errors = strategy_manager.validate_strategy(invalid_dict)
        assert len(errors) > 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_strategy(self):
        """Create a sample strategy configuration."""
        return StrategyConfig(
            name="Sample Strategy",
            pattern_id="bear_flag",
            min_confidence=0.85
        )
    
    def test_save_strategy_config(self, temp_dir, sample_strategy):
        """Test save_strategy_config convenience function."""
        file_path = save_strategy_config(sample_strategy, strategies_dir=temp_dir)
        
        assert os.path.exists(file_path)
        assert file_path.startswith(temp_dir)
    
    def test_load_strategy_config(self, temp_dir, sample_strategy):
        """Test load_strategy_config convenience function."""
        # Save strategy first
        file_path = save_strategy_config(sample_strategy, strategies_dir=temp_dir)
        filename = os.path.basename(file_path)
        
        # Load strategy back
        loaded_strategy = load_strategy_config(filename, strategies_dir=temp_dir)
        
        assert loaded_strategy.name == sample_strategy.name
        assert loaded_strategy.pattern_id == sample_strategy.pattern_id
    
    def test_create_example_strategies(self, temp_dir):
        """Test create_example_strategies function."""
        manager = StrategyManager(strategies_dir=temp_dir)
        
        created_files = create_example_strategies(manager, pattern_id="test_pattern")
        
        assert len(created_files) > 0
        for file_path in created_files:
            assert os.path.exists(file_path)
            assert file_path.endswith('.yaml')


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_save_strategy_io_error(self, temp_dir):
        """Test save_strategy with I/O error."""
        manager = StrategyManager(strategies_dir=temp_dir)
        strategy = StrategyConfig(
            name="Test Strategy",
            pattern_id="bear_flag",
            min_confidence=0.85
        )
        
        # Mock file writing to raise an exception
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(StrategyError, match="Failed to save strategy"):
                manager.save_strategy(strategy)
    
    def test_load_strategy_corrupted_yaml(self, temp_dir):
        """Test loading strategy with corrupted YAML."""
        manager = StrategyManager(strategies_dir=temp_dir)
        
        # Create a corrupted YAML file
        corrupted_file = Path(temp_dir) / "corrupted.yaml"
        with open(corrupted_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(StrategyError, match="Failed to load strategy"):
            manager.load_strategy("corrupted.yaml")
    
    def test_load_strategy_invalid_config(self, temp_dir):
        """Test loading strategy with invalid configuration."""
        manager = StrategyManager(strategies_dir=temp_dir)
        
        # Create YAML with invalid strategy config
        invalid_file = Path(temp_dir) / "invalid.yaml"
        invalid_config = {
            'name': 'Test Strategy',
            'pattern_id': 'bear_flag',
            'min_confidence': 1.5  # Invalid confidence
        }
        
        with open(invalid_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(StrategyError, match="Failed to load strategy"):
            manager.load_strategy("invalid.yaml")
    
    def test_list_strategies_with_corrupted_file(self, temp_dir):
        """Test list_strategies with corrupted file in directory."""
        manager = StrategyManager(strategies_dir=temp_dir)
        
        # Create a valid strategy
        valid_strategy = StrategyConfig(
            name="Valid Strategy",
            pattern_id="bear_flag",
            min_confidence=0.85
        )
        manager.save_strategy(valid_strategy)
        
        # Create a corrupted file
        corrupted_file = Path(temp_dir) / "corrupted.yaml"
        with open(corrupted_file, 'w') as f:
            f.write("invalid yaml content")
        
        # Should handle corrupted file gracefully and return valid strategies
        strategies = manager.list_strategies()
        assert len(strategies) >= 1  # Should include the valid strategy
        
        # Verify the valid strategy is in the list
        valid_found = any(s['name'] == 'Valid Strategy' for s in strategies)
        assert valid_found


if __name__ == "__main__":
    pytest.main([__file__]) 