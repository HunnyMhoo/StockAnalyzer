"""
Pattern Labeler for Stock Trading Patterns

This module provides functionality to manually label historical stock chart segments
that match preferred trading patterns. Labels are stored in JSON format for future
machine learning training.
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Handle imports for both direct execution and package usage
try:
    from .data_fetcher import fetch_hk_stocks, _validate_hk_ticker
except ImportError:
    from data_fetcher import fetch_hk_stocks, _validate_hk_ticker


# Configuration constants
LABELS_DIR = "labels"
DEFAULT_LABELS_FILE = "labeled_patterns.json"
DATE_FORMAT = "%Y-%m-%d"
HK_TICKER_PATTERN = r"^\d{4}\.HK$"


class PatternLabelError(Exception):
    """Custom exception for pattern labeling errors."""
    pass


class ValidationError(PatternLabelError):
    """Exception raised for validation errors."""
    pass


@dataclass
class PatternLabel:
    """
    Data class representing a labeled stock pattern.
    
    Attributes:
        ticker: Stock ticker (e.g., '0700.HK')
        start_date: Pattern start date in YYYY-MM-DD format
        end_date: Pattern end date in YYYY-MM-DD format
        label_type: Type of pattern (default: 'positive')
        notes: Optional notes about the pattern
        created_at: Timestamp when label was created
    """
    ticker: str
    start_date: str
    end_date: str
    label_type: str = "positive"
    notes: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        """Set created_at timestamp if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class LabelValidator:
    """Validator for pattern labels with comprehensive validation rules."""
    
    @staticmethod
    def validate_ticker(ticker: str) -> None:
        """
        Validate Hong Kong stock ticker format.
        
        Args:
            ticker: Stock ticker string
            
        Raises:
            ValidationError: If ticker format is invalid
        """
        if not ticker or not isinstance(ticker, str):
            raise ValidationError("Ticker must be a non-empty string")
            
        if not _validate_hk_ticker(ticker):
            raise ValidationError(
                f"Invalid ticker format: {ticker}. "
                f"Expected format: XXXX.HK (e.g., '0700.HK')"
            )
    
    @staticmethod
    def validate_date_format(date_str: str, field_name: str) -> datetime:
        """
        Validate date format and convert to datetime object.
        
        Args:
            date_str: Date string to validate
            field_name: Name of the field for error messages
            
        Returns:
            datetime: Parsed datetime object
            
        Raises:
            ValidationError: If date format is invalid
        """
        if date_str is None or not isinstance(date_str, str) or not date_str:
            raise ValidationError(f"{field_name} must be a non-empty string")
            
        try:
            return datetime.strptime(date_str, DATE_FORMAT)
        except ValueError:
            raise ValidationError(
                f"Invalid {field_name} format: {date_str}. "
                f"Expected format: YYYY-MM-DD"
            )
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> None:
        """
        Validate that start_date is before end_date.
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Raises:
            ValidationError: If date range is invalid
        """
        start_dt = LabelValidator.validate_date_format(start_date, "start_date")
        end_dt = LabelValidator.validate_date_format(end_date, "end_date")
        
        if start_dt >= end_dt:
            raise ValidationError(
                f"start_date ({start_date}) must be before end_date ({end_date})"
            )
    
    @staticmethod
    def validate_label_type(label_type: str) -> None:
        """
        Validate label type.
        
        Args:
            label_type: Label type string
            
        Raises:
            ValidationError: If label type is invalid
        """
        valid_types = {"positive", "negative", "neutral"}
        if label_type not in valid_types:
            raise ValidationError(
                f"Invalid label_type: {label_type}. "
                f"Valid types: {', '.join(valid_types)}"
            )
    
    @classmethod
    def validate_pattern_label(cls, label: PatternLabel) -> None:
        """
        Validate all aspects of a pattern label.
        
        Args:
            label: PatternLabel instance to validate
            
        Raises:
            ValidationError: If any validation fails
        """
        cls.validate_ticker(label.ticker)
        cls.validate_date_range(label.start_date, label.end_date)
        cls.validate_label_type(label.label_type)


class PatternLabeler:
    """
    Main class for managing pattern labels with JSON persistence.
    """
    
    def __init__(self, labels_file: Optional[str] = None):
        """
        Initialize PatternLabeler.
        
        Args:
            labels_file: Path to labels JSON file (optional)
        """
        self.labels_file = labels_file or os.path.join(LABELS_DIR, DEFAULT_LABELS_FILE)
        self._ensure_labels_directory()
        self._labels_cache: List[PatternLabel] = []
        self._cache_dirty = False
    
    def _ensure_labels_directory(self) -> None:
        """Create labels directory if it doesn't exist."""
        labels_dir = os.path.dirname(self.labels_file)
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
            print(f"✓ Created labels directory: {labels_dir}/")
    
    def _get_label_key(self, label: PatternLabel) -> str:
        """
        Generate unique key for a pattern label.
        
        Args:
            label: PatternLabel instance
            
        Returns:
            str: Unique key for the label
        """
        return f"{label.ticker}_{label.start_date}_{label.end_date}"
    
    def add_label(self, 
                  ticker: str,
                  start_date: str, 
                  end_date: str,
                  label_type: str = "positive",
                  notes: str = "",
                  overwrite: bool = False) -> PatternLabel:
        """
        Add a new pattern label.
        
        Args:
            ticker: Stock ticker
            start_date: Pattern start date (YYYY-MM-DD)
            end_date: Pattern end date (YYYY-MM-DD)
            label_type: Type of pattern (default: 'positive')
            notes: Optional notes
            overwrite: Whether to overwrite existing labels
            
        Returns:
            PatternLabel: The created label
            
        Raises:
            ValidationError: If validation fails
            PatternLabelError: If duplicate exists and overwrite=False
        """
        # Create and validate label
        label = PatternLabel(
            ticker=ticker.upper(),
            start_date=start_date,
            end_date=end_date,
            label_type=label_type,
            notes=notes
        )
        
        LabelValidator.validate_pattern_label(label)
        
        # Check for duplicates
        existing_labels = self.load_labels()
        label_key = self._get_label_key(label)
        
        for existing_label in existing_labels:
            if self._get_label_key(existing_label) == label_key:
                if not overwrite:
                    raise PatternLabelError(
                        f"Label already exists for {ticker} "
                        f"({start_date} to {end_date}). "
                        f"Use overwrite=True to replace it."
                    )
                else:
                    existing_labels.remove(existing_label)
                    break
        
        existing_labels.append(label)
        self.save_labels(existing_labels)
        
        print(f"✓ Added pattern label: {ticker} ({start_date} to {end_date})")
        return label
    
    def load_labels(self) -> List[PatternLabel]:
        """
        Load pattern labels from JSON file.
        
        Returns:
            List[PatternLabel]: List of loaded labels
            
        Raises:
            PatternLabelError: If loading fails
        """
        if not os.path.exists(self.labels_file):
            return []
        
        try:
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            labels = []
            for item in data:
                labels.append(PatternLabel(**item))
            
            return labels
            
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            raise PatternLabelError(f"Failed to load labels from {self.labels_file}: {e}")
    
    def save_labels(self, labels: List[PatternLabel]) -> None:
        """
        Save pattern labels to JSON file.
        
        Args:
            labels: List of PatternLabel instances to save
            
        Raises:
            PatternLabelError: If saving fails
        """
        try:
            # Convert to dictionaries for JSON serialization
            data = [asdict(label) for label in labels]
            
            # Write atomically by using a temporary file
            temp_file = self.labels_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            os.rename(temp_file, self.labels_file)
            
            print(f"✓ Saved {len(labels)} pattern labels to {self.labels_file}")
            
        except (IOError, OSError) as e:
            raise PatternLabelError(f"Failed to save labels to {self.labels_file}: {e}")
    
    def get_labels_by_ticker(self, ticker: str) -> List[PatternLabel]:
        """
        Get all labels for a specific ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            List[PatternLabel]: Labels for the ticker
        """
        labels = self.load_labels()
        return [label for label in labels if label.ticker.upper() == ticker.upper()]
    
    def get_labels_summary(self) -> Dict[str, int]:
        """
        Get summary statistics of labeled patterns.
        
        Returns:
            Dict[str, int]: Summary statistics
        """
        labels = self.load_labels()
        
        summary = {
            "total_labels": len(labels),
            "unique_tickers": len(set(label.ticker for label in labels)),
            "positive_labels": sum(1 for label in labels if label.label_type == "positive"),
            "negative_labels": sum(1 for label in labels if label.label_type == "negative"),
            "neutral_labels": sum(1 for label in labels if label.label_type == "neutral"),
        }
        
        return summary
    
    def remove_label(self, ticker: str, start_date: str, end_date: str) -> bool:
        """
        Remove a specific pattern label.
        
        Args:
            ticker: Stock ticker
            start_date: Pattern start date
            end_date: Pattern end date
            
        Returns:
            bool: True if label was removed, False if not found
        """
        labels = self.load_labels()
        label_key = f"{ticker.upper()}_{start_date}_{end_date}"
        
        for i, label in enumerate(labels):
            if self._get_label_key(label) == label_key:
                removed_label = labels.pop(i)
                self.save_labels(labels)
                print(f"✓ Removed pattern label: {removed_label.ticker} "
                      f"({removed_label.start_date} to {removed_label.end_date})")
                return True
        
        print(f"⚠️  Label not found: {ticker} ({start_date} to {end_date})")
        return False


# Convenience functions for notebook usage
def save_labeled_patterns(label_list: List[Dict], path: Optional[str] = None) -> None:
    """
    Save a list of pattern labels to JSON file.
    
    Args:
        label_list: List of label dictionaries
        path: Optional path to save file (default: labels/labeled_patterns.json)
        
    Raises:
        ValidationError: If any label is invalid
        PatternLabelError: If saving fails
    """
    labeler = PatternLabeler(path)
    
    # Convert dictionaries to PatternLabel instances and validate
    labels = []
    for item in label_list:
        # Handle both dict and PatternLabel inputs
        if isinstance(item, PatternLabel):
            label = item
        else:
            label = PatternLabel(**item)
        
        LabelValidator.validate_pattern_label(label)
        labels.append(label)
    
    labeler.save_labels(labels)


def load_labeled_patterns(path: Optional[str] = None) -> List[PatternLabel]:
    """
    Load pattern labels from JSON file.
    
    Args:
        path: Optional path to load file (default: labels/labeled_patterns.json)
        
    Returns:
        List[PatternLabel]: List of loaded labels
        
    Raises:
        PatternLabelError: If loading fails
    """
    labeler = PatternLabeler(path)
    return labeler.load_labels() 