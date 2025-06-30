"""
Unit tests for pattern labeling functionality.

Tests validation, JSON persistence, error handling, and all public methods
of the pattern labeling system.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch, mock_open

import pytest

# Import the modules to test
from stock_analyzer.patterns import (
    PatternLabel,
    LabelValidator,
    PatternLabeler,
    save_labeled_patterns,
    load_labeled_patterns,
    ValidationError,
    PatternLabelError
)


class TestPatternLabel(unittest.TestCase):
    """Test the PatternLabel dataclass."""
    
    def test_pattern_label_creation(self):
        """Test basic PatternLabel creation."""
        label = PatternLabel(
            ticker="0700.HK",
            start_date="2023-01-01",
            end_date="2023-01-15",
            label_type="positive",
            notes="Test pattern"
        )
        
        self.assertEqual(label.ticker, "0700.HK")
        self.assertEqual(label.start_date, "2023-01-01")
        self.assertEqual(label.end_date, "2023-01-15")
        self.assertEqual(label.label_type, "positive")
        self.assertEqual(label.notes, "Test pattern")
        self.assertIsNotNone(label.created_at)
    
    def test_pattern_label_defaults(self):
        """Test PatternLabel with default values."""
        label = PatternLabel(
            ticker="0700.HK",
            start_date="2023-01-01",
            end_date="2023-01-15"
        )
        
        self.assertEqual(label.label_type, "positive")
        self.assertEqual(label.notes, "")
        self.assertIsNotNone(label.created_at)
    
    def test_created_at_auto_generation(self):
        """Test that created_at is automatically generated."""
        label = PatternLabel(
            ticker="0700.HK",
            start_date="2023-01-01",
            end_date="2023-01-15"
        )
        
        # Should be a valid ISO format timestamp
        created_time = datetime.fromisoformat(label.created_at)
        self.assertIsInstance(created_time, datetime)


class TestLabelValidator(unittest.TestCase):
    """Test the LabelValidator class."""
    
    def test_validate_ticker_valid(self):
        """Test validation of valid HK tickers."""
        valid_tickers = ["0700.HK", "0001.HK", "9999.HK"]
        
        for ticker in valid_tickers:
            LabelValidator.validate_ticker(ticker)  # Should not raise
    
    def test_validate_ticker_invalid(self):
        """Test validation of invalid tickers."""
        invalid_tickers = [
            "AAPL",         # US stock
            "700.HK",       # Missing leading zero
            "0700.HKX",     # Wrong suffix
            "0700",         # Missing .HK
            "",             # Empty string
            None,           # None value
            123             # Non-string
        ]
        
        for ticker in invalid_tickers:
            with self.assertRaises(ValidationError):
                LabelValidator.validate_ticker(ticker)
    
    def test_validate_date_format_valid(self):
        """Test validation of valid date formats."""
        valid_dates = ["2023-01-01", "2022-12-31", "2024-02-29"]  # Leap year
        
        for date_str in valid_dates:
            result = LabelValidator.validate_date_format(date_str, "test_date")
            self.assertIsInstance(result, datetime)
    
    def test_validate_date_format_invalid(self):
        """Test validation of invalid date formats."""
        invalid_dates = [
            ("01/01/2023", "Wrong format"),
            ("2023-13-01", "Invalid month"),
            ("2023-01-32", "Invalid day"),
            ("", "Empty string"),
            (None, "None value"),
            ("not-a-date", "Invalid format"),
            ("2023/01/01", "Wrong separator"),
            ("2023-1", "Incomplete date")
        ]
        
        for date_str, description in invalid_dates:
            with self.assertRaises(ValidationError, msg=f"Should fail for {description}: {date_str}"):
                LabelValidator.validate_date_format(date_str, "test_date")
    
    def test_validate_date_range_valid(self):
        """Test validation of valid date ranges."""
        valid_ranges = [
            ("2023-01-01", "2023-01-02"),
            ("2023-01-01", "2023-12-31"),
            ("2022-12-31", "2023-01-01")
        ]
        
        for start, end in valid_ranges:
            LabelValidator.validate_date_range(start, end)  # Should not raise
    
    def test_validate_date_range_invalid(self):
        """Test validation of invalid date ranges."""
        invalid_ranges = [
            ("2023-01-02", "2023-01-01"),  # Start after end
            ("2023-01-01", "2023-01-01"),  # Same date
        ]
        
        for start, end in invalid_ranges:
            with self.assertRaises(ValidationError):
                LabelValidator.validate_date_range(start, end)
    
    def test_validate_label_type_valid(self):
        """Test validation of valid label types."""
        valid_types = ["positive", "negative", "neutral"]
        
        for label_type in valid_types:
            LabelValidator.validate_label_type(label_type)  # Should not raise
    
    def test_validate_label_type_invalid(self):
        """Test validation of invalid label types."""
        invalid_types = ["good", "bad", "maybe", "", None]
        
        for label_type in invalid_types:
            with self.assertRaises(ValidationError):
                LabelValidator.validate_label_type(label_type)
    
    def test_validate_pattern_label_valid(self):
        """Test validation of valid pattern labels."""
        valid_label = PatternLabel(
            ticker="0700.HK",
            start_date="2023-01-01",
            end_date="2023-01-15",
            label_type="positive"
        )
        
        LabelValidator.validate_pattern_label(valid_label)  # Should not raise
    
    def test_validate_pattern_label_invalid(self):
        """Test validation of invalid pattern labels."""
        # Invalid ticker
        invalid_label = PatternLabel(
            ticker="INVALID",
            start_date="2023-01-01",
            end_date="2023-01-15",
            label_type="positive"
        )
        
        with self.assertRaises(ValidationError):
            LabelValidator.validate_pattern_label(invalid_label)


class TestPatternLabeler(unittest.TestCase):
    """Test the PatternLabeler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.labels_file = os.path.join(self.temp_dir, "test_labels.json")
        self.labeler = PatternLabeler(self.labels_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove test files
        if os.path.exists(self.labels_file):
            os.remove(self.labels_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """Test PatternLabeler initialization."""
        self.assertEqual(self.labeler.labels_file, self.labels_file)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_add_label_valid(self):
        """Test adding valid labels."""
        label = self.labeler.add_label(
            ticker="0700.HK",
            start_date="2023-01-01",
            end_date="2023-01-15",
            label_type="positive",
            notes="Test pattern"
        )
        
        self.assertIsInstance(label, PatternLabel)
        self.assertEqual(label.ticker, "0700.HK")
        self.assertTrue(os.path.exists(self.labels_file))
    
    def test_add_label_invalid(self):
        """Test adding invalid labels."""
        with self.assertRaises(ValidationError):
            self.labeler.add_label(
                ticker="INVALID",
                start_date="2023-01-01",
                end_date="2023-01-15"
            )
    
    def test_add_duplicate_label_no_overwrite(self):
        """Test adding duplicate labels without overwrite."""
        # Add first label
        self.labeler.add_label(
            ticker="0700.HK",
            start_date="2023-01-01",
            end_date="2023-01-15"
        )
        
        # Try to add duplicate
        with self.assertRaises(PatternLabelError):
            self.labeler.add_label(
                ticker="0700.HK",
                start_date="2023-01-01",
                end_date="2023-01-15",
                overwrite=False
            )
    
    def test_add_duplicate_label_with_overwrite(self):
        """Test adding duplicate labels with overwrite."""
        # Add first label
        self.labeler.add_label(
            ticker="0700.HK",
            start_date="2023-01-01",
            end_date="2023-01-15",
            notes="Original"
        )
        
        # Overwrite with new label
        updated_label = self.labeler.add_label(
            ticker="0700.HK",
            start_date="2023-01-01",
            end_date="2023-01-15",
            notes="Updated",
            overwrite=True
        )
        
        self.assertEqual(updated_label.notes, "Updated")
    
    def test_save_and_load_labels(self):
        """Test saving and loading labels."""
        # Create test labels
        labels = [
            PatternLabel("0700.HK", "2023-01-01", "2023-01-15", "positive", "Test 1"),
            PatternLabel("0001.HK", "2023-02-01", "2023-02-15", "negative", "Test 2")
        ]
        
        # Save labels
        self.labeler.save_labels(labels)
        
        # Load labels
        loaded_labels = self.labeler.load_labels()
        
        self.assertEqual(len(loaded_labels), 2)
        self.assertEqual(loaded_labels[0].ticker, "0700.HK")
        self.assertEqual(loaded_labels[1].ticker, "0001.HK")
    
    def test_load_labels_empty_file(self):
        """Test loading labels from non-existent file."""
        non_existent_file = os.path.join(self.temp_dir, "non_existent.json")
        labeler = PatternLabeler(non_existent_file)
        
        labels = labeler.load_labels()
        self.assertEqual(labels, [])
    
    def test_get_labels_by_ticker(self):
        """Test filtering labels by ticker."""
        # Add multiple labels
        self.labeler.add_label("0700.HK", "2023-01-01", "2023-01-15")
        self.labeler.add_label("0001.HK", "2023-02-01", "2023-02-15")
        self.labeler.add_label("0700.HK", "2023-03-01", "2023-03-15")
        
        # Get labels for specific ticker
        hk700_labels = self.labeler.get_labels_by_ticker("0700.HK")
        
        self.assertEqual(len(hk700_labels), 2)
        for label in hk700_labels:
            self.assertEqual(label.ticker, "0700.HK")
    
    def test_get_labels_summary(self):
        """Test getting summary statistics."""
        # Add test labels
        self.labeler.add_label("0700.HK", "2023-01-01", "2023-01-15", "positive")
        self.labeler.add_label("0001.HK", "2023-02-01", "2023-02-15", "negative")
        self.labeler.add_label("0005.HK", "2023-03-01", "2023-03-15", "positive")
        
        summary = self.labeler.get_labels_summary()
        
        self.assertEqual(summary["total_labels"], 3)
        self.assertEqual(summary["unique_tickers"], 3)
        self.assertEqual(summary["positive_labels"], 2)
        self.assertEqual(summary["negative_labels"], 1)
        self.assertEqual(summary["neutral_labels"], 0)
    
    def test_remove_label_existing(self):
        """Test removing existing label."""
        # Add a label
        self.labeler.add_label("0700.HK", "2023-01-01", "2023-01-15")
        
        # Remove the label
        removed = self.labeler.remove_label("0700.HK", "2023-01-01", "2023-01-15")
        
        self.assertTrue(removed)
        
        # Verify it's gone
        labels = self.labeler.load_labels()
        self.assertEqual(len(labels), 0)
    
    def test_remove_label_non_existent(self):
        """Test removing non-existent label."""
        removed = self.labeler.remove_label("0700.HK", "2023-01-01", "2023-01-15")
        self.assertFalse(removed)


class TestConvenienceFunctions(unittest.TestCase):
    """Test the convenience functions for notebook usage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.labels_file = os.path.join(self.temp_dir, "convenience_test.json")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.labels_file):
            os.remove(self.labels_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_save_labeled_patterns_dict_format(self):
        """Test saving patterns in dictionary format."""
        patterns = [
            {
                "ticker": "0700.HK",
                "start_date": "2023-01-01",
                "end_date": "2023-01-15",
                "label_type": "positive",
                "notes": "Test pattern"
            }
        ]
        
        save_labeled_patterns(patterns, self.labels_file)
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.labels_file))
        
        # Verify content
        with open(self.labels_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["ticker"], "0700.HK")
    
    def test_save_labeled_patterns_invalid(self):
        """Test saving invalid patterns."""
        invalid_patterns = [
            {
                "ticker": "INVALID",
                "start_date": "2023-01-01",
                "end_date": "2023-01-15"
            }
        ]
        
        with self.assertRaises(ValidationError):
            save_labeled_patterns(invalid_patterns, self.labels_file)
    
    def test_load_labeled_patterns(self):
        """Test loading patterns using convenience function."""
        # First save some patterns
        patterns = [
            {
                "ticker": "0700.HK",
                "start_date": "2023-01-01",
                "end_date": "2023-01-15",
                "label_type": "positive"
            }
        ]
        
        save_labeled_patterns(patterns, self.labels_file)
        
        # Load using convenience function
        loaded_patterns = load_labeled_patterns(self.labels_file)
        
        self.assertEqual(len(loaded_patterns), 1)
        self.assertEqual(loaded_patterns[0].ticker, "0700.HK")
        self.assertIsInstance(loaded_patterns[0], PatternLabel)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    def test_json_corruption_handling(self):
        """Test handling of corrupted JSON files."""
        temp_dir = tempfile.mkdtemp()
        corrupted_file = os.path.join(temp_dir, "corrupted.json")
        
        # Create corrupted JSON file
        with open(corrupted_file, 'w') as f:
            f.write("{ invalid json content")
        
        labeler = PatternLabeler(corrupted_file)
        
        with self.assertRaises(PatternLabelError):
            labeler.load_labels()
        
        # Cleanup
        os.remove(corrupted_file)
        os.rmdir(temp_dir)
    
    def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        # This test is platform-dependent and may not work on all systems
        # It's included for completeness but may be skipped in some environments
        pass


if __name__ == '__main__':
    unittest.main() 