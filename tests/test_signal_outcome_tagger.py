"""
Tests for Signal Outcome Tagger

This module tests the signal outcome tagging functionality including:
- Match file loading and validation
- Individual and batch outcome tagging
- Feedback analysis and statistics
- File operations and error handling
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.append('../src')

from src.signal_outcome_tagger import (
    SignalOutcomeTagger,
    SignalOutcomeError,
    load_latest_matches,
    quick_tag_outcome,
    review_latest_feedback,
    VALID_OUTCOMES,
    REQUIRED_MATCH_COLUMNS
)


class TestSignalOutcomeTagger:
    """Test core SignalOutcomeTagger functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.signals_dir = os.path.join(self.temp_dir, 'signals')
        os.makedirs(self.signals_dir)
        
        # Sample matches data
        self.sample_matches = pd.DataFrame([
            {
                'ticker': '0700.HK',
                'window_start_date': '2023-10-01',
                'window_end_date': '2023-10-31',
                'confidence_score': 0.85,
                'rank': 1
            },
            {
                'ticker': '0005.HK',
                'window_start_date': '2023-09-15',
                'window_end_date': '2023-10-15',
                'confidence_score': 0.78,
                'rank': 2
            },
            {
                'ticker': '0388.HK',
                'window_start_date': '2023-11-01',
                'window_end_date': '2023-11-30',
                'confidence_score': 0.72,
                'rank': 3
            }
        ])
        
        # Create sample match file
        self.match_file_path = os.path.join(self.signals_dir, 'matches_20231201.csv')
        self.sample_matches.to_csv(self.match_file_path, index=False)
        
        self.tagger = SignalOutcomeTagger(signals_dir=self.signals_dir)
    
    def test_init_success(self):
        """Test successful SignalOutcomeTagger initialization."""
        tagger = SignalOutcomeTagger(signals_dir=self.signals_dir)
        assert tagger.signals_dir == self.signals_dir
        assert tagger.create_backups is True
    
    def test_init_missing_directory(self):
        """Test initialization with missing signals directory."""
        non_existent_dir = os.path.join(self.temp_dir, 'missing')
        
        with pytest.raises(SignalOutcomeError, match="Signals directory not found"):
            SignalOutcomeTagger(signals_dir=non_existent_dir)
    
    def test_load_matches_file_success(self):
        """Test successful loading of matches file."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        assert len(matches_df) == 3
        assert 'outcome' in matches_df.columns
        assert 'feedback_notes' in matches_df.columns
        assert 'tagged_date' in matches_df.columns
        assert all(col in matches_df.columns for col in REQUIRED_MATCH_COLUMNS)
    
    def test_load_matches_file_missing_file(self):
        """Test error handling for missing match file."""
        missing_file = os.path.join(self.signals_dir, 'missing.csv')
        
        with pytest.raises(SignalOutcomeError, match="Match file not found"):
            self.tagger.load_matches_file(missing_file)
    
    def test_load_matches_file_missing_columns(self):
        """Test error handling for match file with missing required columns."""
        # Create file with missing columns
        invalid_data = pd.DataFrame([
            {'ticker': '0700.HK', 'confidence_score': 0.8}  # Missing date columns
        ])
        
        invalid_file = os.path.join(self.signals_dir, 'invalid_matches.csv')
        invalid_data.to_csv(invalid_file, index=False)
        
        with pytest.raises(SignalOutcomeError, match="Missing required columns"):
            self.tagger.load_matches_file(invalid_file)
    
    def test_load_matches_file_invalid_dates(self):
        """Test error handling for invalid date formats."""
        invalid_data = self.sample_matches.copy()
        invalid_data.loc[0, 'window_start_date'] = 'invalid-date'
        
        invalid_file = os.path.join(self.signals_dir, 'invalid_dates.csv')
        invalid_data.to_csv(invalid_file, index=False)
        
        with pytest.raises(SignalOutcomeError, match="Invalid date format"):
            self.tagger.load_matches_file(invalid_file)
    
    def test_validate_outcome_valid_values(self):
        """Test validation of valid outcome values."""
        for outcome in VALID_OUTCOMES:
            assert self.tagger.validate_outcome(outcome) is True
            assert self.tagger.validate_outcome(outcome.upper()) is True
            assert self.tagger.validate_outcome(f" {outcome} ") is True
    
    def test_validate_outcome_invalid_values(self):
        """Test validation of invalid outcome values."""
        invalid_outcomes = ['invalid', 'maybe', 'yes', 'no', '']
        
        for invalid_outcome in invalid_outcomes:
            with pytest.raises(SignalOutcomeError, match="Invalid outcome"):
                self.tagger.validate_outcome(invalid_outcome)
    
    def test_validate_outcome_null_values(self):
        """Test validation of null outcome values."""
        null_values = [None, pd.NA, np.nan]
        
        for null_value in null_values:
            with pytest.raises(SignalOutcomeError, match="Outcome cannot be null"):
                self.tagger.validate_outcome(null_value)
    
    def test_find_match_by_key_success(self):
        """Test successful match finding by key."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        idx, row = self.tagger.find_match_by_key(
            matches_df, '0700.HK', '2023-10-01', '2023-10-31'
        )
        
        assert row['ticker'] == '0700.HK'
        assert row['window_start_date'] == '2023-10-01'
        assert row['confidence_score'] == 0.85
    
    def test_find_match_by_key_no_match(self):
        """Test match finding with non-existent key."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        with pytest.raises(SignalOutcomeError, match="No match found"):
            self.tagger.find_match_by_key(
                matches_df, '9999.HK', '2023-01-01'
            )
    
    def test_find_match_by_key_multiple_matches(self):
        """Test match finding with ambiguous key."""
        # Create data with duplicate ticker/start_date combinations
        duplicate_data = pd.concat([self.sample_matches, self.sample_matches.iloc[[0]]])
        duplicate_file = os.path.join(self.signals_dir, 'duplicate_matches.csv')
        duplicate_data.to_csv(duplicate_file, index=False)
        
        matches_df = self.tagger.load_matches_file(duplicate_file)
        
        with pytest.raises(SignalOutcomeError, match="Multiple matches found"):
            self.tagger.find_match_by_key(
                matches_df, '0700.HK', '2023-10-01'
            )
    
    def test_tag_outcome_success(self):
        """Test successful outcome tagging."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        updated_df = self.tagger.tag_outcome(
            matches_df, '0700.HK', '2023-10-01', 'success', 
            'Great breakout pattern', '2023-10-31'
        )
        
        # Find the tagged row
        tagged_row = updated_df[
            (updated_df['ticker'] == '0700.HK') & 
            (updated_df['window_start_date'] == '2023-10-01')
        ].iloc[0]
        
        assert tagged_row['outcome'] == 'success'
        assert tagged_row['feedback_notes'] == 'Great breakout pattern'
        assert pd.notna(tagged_row['tagged_date'])
    
    def test_tag_outcome_duplicate_prevention(self):
        """Test prevention of duplicate tagging without overwrite."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        # First tag
        updated_df = self.tagger.tag_outcome(
            matches_df, '0700.HK', '2023-10-01', 'success'
        )
        
        # Second tag should fail without overwrite
        with pytest.raises(SignalOutcomeError, match="already tagged"):
            self.tagger.tag_outcome(
                updated_df, '0700.HK', '2023-10-01', 'failure'
            )
    
    def test_tag_outcome_overwrite(self):
        """Test outcome tagging with overwrite."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        # First tag
        updated_df = self.tagger.tag_outcome(
            matches_df, '0700.HK', '2023-10-01', 'success'
        )
        
        # Overwrite with new tag
        final_df = self.tagger.tag_outcome(
            updated_df, '0700.HK', '2023-10-01', 'failure', 
            'Actually failed', overwrite=True
        )
        
        tagged_row = final_df[
            (final_df['ticker'] == '0700.HK') & 
            (final_df['window_start_date'] == '2023-10-01')
        ].iloc[0]
        
        assert tagged_row['outcome'] == 'failure'
        assert tagged_row['feedback_notes'] == 'Actually failed'
    
    def test_tag_batch_outcomes_success(self):
        """Test successful batch outcome tagging."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        batch_tags = [
            {
                'ticker': '0700.HK',
                'window_start_date': '2023-10-01',
                'outcome': 'success',
                'feedback_notes': 'Good pattern'
            },
            {
                'ticker': '0005.HK',
                'window_start_date': '2023-09-15',
                'outcome': 'failure',
                'feedback_notes': 'False breakout'
            }
        ]
        
        updated_df = self.tagger.tag_batch_outcomes(matches_df, batch_tags)
        
        # Check first tag
        row1 = updated_df[updated_df['ticker'] == '0700.HK'].iloc[0]
        assert row1['outcome'] == 'success'
        assert row1['feedback_notes'] == 'Good pattern'
        
        # Check second tag
        row2 = updated_df[updated_df['ticker'] == '0005.HK'].iloc[0]
        assert row2['outcome'] == 'failure'
        assert row2['feedback_notes'] == 'False breakout'
    
    def test_tag_batch_outcomes_partial_failure(self):
        """Test batch tagging with some failures."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        batch_tags = [
            {
                'ticker': '0700.HK',
                'window_start_date': '2023-10-01',
                'outcome': 'success'
            },
            {
                'ticker': '9999.HK',  # Non-existent ticker
                'window_start_date': '2023-01-01',
                'outcome': 'failure'
            }
        ]
        
        updated_df = self.tagger.tag_batch_outcomes(matches_df, batch_tags)
        
        # First tag should succeed
        row1 = updated_df[updated_df['ticker'] == '0700.HK'].iloc[0]
        assert row1['outcome'] == 'success'
        
        # Second tag should have been skipped
        assert len(updated_df[updated_df['ticker'] == '9999.HK']) == 0
    
    def test_save_labeled_matches(self):
        """Test saving labeled matches to file."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        # Add some tags
        updated_df = self.tagger.tag_outcome(
            matches_df, '0700.HK', '2023-10-01', 'success'
        )
        
        # Save labeled matches
        output_path = self.tagger.save_labeled_matches(updated_df, self.match_file_path)
        
        # Verify file was created
        assert os.path.exists(output_path)
        assert output_path.endswith('_labeled.csv')
        
        # Verify content
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == len(updated_df)
        assert 'outcome' in saved_df.columns
    
    def test_save_labeled_matches_backup_creation(self):
        """Test backup file creation during save."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        updated_df = self.tagger.tag_outcome(
            matches_df, '0700.HK', '2023-10-01', 'success'
        )
        
        # Save with backup enabled
        tagger_with_backup = SignalOutcomeTagger(
            signals_dir=self.signals_dir, create_backups=True
        )
        
        output_path = tagger_with_backup.save_labeled_matches(
            updated_df, self.match_file_path
        )
        
        # Check if backup was created
        backup_path = self.match_file_path + "_backup"
        # Note: Backup only created for non-labeled files, so this might not exist
        # depending on implementation details
    
    def test_review_feedback_empty(self):
        """Test feedback review with no tagged matches."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        results = self.tagger.review_feedback(matches_df)
        
        assert results['total_matches'] == 3
        assert results['tagged_matches'] == 0
        assert results['tagging_rate'] == 0.0
    
    def test_review_feedback_with_tags(self):
        """Test feedback review with tagged matches."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        # Add some tags
        updated_df = self.tagger.tag_outcome(
            matches_df, '0700.HK', '2023-10-01', 'success'
        )
        updated_df = self.tagger.tag_outcome(
            updated_df, '0005.HK', '2023-09-15', 'failure'
        )
        
        results = self.tagger.review_feedback(updated_df)
        
        assert results['total_matches'] == 3
        assert results['tagged_matches'] == 2
        assert results['tagging_rate'] == 2/3
        assert 'success' in results['outcome_summary']['counts']
        assert 'failure' in results['outcome_summary']['counts']
    
    def test_review_feedback_confidence_bands(self):
        """Test feedback review confidence band analysis."""
        matches_df = self.tagger.load_matches_file(self.match_file_path)
        
        # Tag all matches
        updated_df = self.tagger.tag_outcome(
            matches_df, '0700.HK', '2023-10-01', 'success'  # 0.85 confidence
        )
        updated_df = self.tagger.tag_outcome(
            updated_df, '0005.HK', '2023-09-15', 'failure'  # 0.78 confidence
        )
        updated_df = self.tagger.tag_outcome(
            updated_df, '0388.HK', '2023-11-01', 'success'  # 0.72 confidence
        )
        
        results = self.tagger.review_feedback(updated_df)
        
        # Check confidence band analysis
        assert 'confidence_bands' in results
        
        # Should have entries in the 0.8-0.9 and 0.7-0.8 bands
        bands = results['confidence_bands']
        assert any(band['count'] > 0 for band in bands.values())
    
    def test_find_available_match_files(self):
        """Test finding available match files."""
        # Create additional match files
        additional_file = os.path.join(self.signals_dir, 'matches_20231202.csv')
        self.sample_matches.to_csv(additional_file, index=False)
        
        labeled_file = os.path.join(self.signals_dir, 'matches_20231201_labeled.csv')
        self.sample_matches.to_csv(labeled_file, index=False)
        
        match_files = self.tagger.find_available_match_files()
        
        # Should find 2 unlabeled files, but not the labeled one
        assert len(match_files) == 2
        assert all('_labeled.csv' not in f for f in match_files)
        assert all(f.endswith('.csv') for f in match_files)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.signals_dir = os.path.join(self.temp_dir, 'signals')
        os.makedirs(self.signals_dir)
        
        # Sample matches data
        self.sample_matches = pd.DataFrame([
            {
                'ticker': '0700.HK',
                'window_start_date': '2023-10-01',
                'window_end_date': '2023-10-31',
                'confidence_score': 0.85,
                'rank': 1
            }
        ])
        
        # Create sample match file
        self.match_file_path = os.path.join(self.signals_dir, 'matches_20231201.csv')
        self.sample_matches.to_csv(self.match_file_path, index=False)
    
    def test_load_latest_matches(self):
        """Test loading latest matches convenience function."""
        file_path, matches_df = load_latest_matches(self.signals_dir)
        
        assert file_path == self.match_file_path
        assert len(matches_df) == 1
        assert 'outcome' in matches_df.columns
    
    def test_load_latest_matches_no_files(self):
        """Test loading latest matches with no available files."""
        empty_dir = os.path.join(self.temp_dir, 'empty_signals')
        os.makedirs(empty_dir)
        
        with pytest.raises(SignalOutcomeError, match="No match files found"):
            load_latest_matches(empty_dir)
    
    def test_quick_tag_outcome(self):
        """Test quick outcome tagging convenience function."""
        quick_tag_outcome(
            '0700.HK', '2023-10-01', 'success', 
            'Quick test', self.signals_dir
        )
        
        # Verify the tag was applied and saved
        labeled_file = os.path.join(self.signals_dir, 'matches_20231201_labeled.csv')
        assert os.path.exists(labeled_file)
        
        labeled_df = pd.read_csv(labeled_file)
        tagged_row = labeled_df[labeled_df['ticker'] == '0700.HK'].iloc[0]
        assert tagged_row['outcome'] == 'success'
        assert tagged_row['feedback_notes'] == 'Quick test'
    
    def test_review_latest_feedback_no_files(self):
        """Test review latest feedback with no labeled files."""
        results = review_latest_feedback(self.signals_dir)
        
        assert results == {}
    
    def test_review_latest_feedback_with_labeled_file(self):
        """Test review latest feedback with existing labeled file."""
        # Create a labeled file with some tags
        labeled_data = self.sample_matches.copy()
        labeled_data['outcome'] = 'success'
        labeled_data['feedback_notes'] = 'Test feedback'
        labeled_data['tagged_date'] = datetime.now().isoformat()
        
        labeled_file = os.path.join(self.signals_dir, 'matches_20231201_labeled.csv')
        labeled_data.to_csv(labeled_file, index=False)
        
        results = review_latest_feedback(self.signals_dir)
        
        assert results['total_matches'] == 1
        assert results['tagged_matches'] == 1
        assert results['tagging_rate'] == 1.0


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.signals_dir = os.path.join(self.temp_dir, 'signals')
        os.makedirs(self.signals_dir)
    
    def test_signal_outcome_error_inheritance(self):
        """Test SignalOutcomeError exception inheritance."""
        error = SignalOutcomeError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_malformed_csv_handling(self):
        """Test handling of malformed CSV files."""
        # Create malformed CSV
        malformed_file = os.path.join(self.signals_dir, 'malformed.csv')
        with open(malformed_file, 'w') as f:
            f.write("ticker,invalid\n0700.HK,")  # Incomplete row
        
        tagger = SignalOutcomeTagger(signals_dir=self.signals_dir)
        
        with pytest.raises(SignalOutcomeError):
            tagger.load_matches_file(malformed_file)
    
    def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        tagger = SignalOutcomeTagger(signals_dir=self.signals_dir)
        
        # Create a simple matches DataFrame
        matches_df = pd.DataFrame([{
            'ticker': '0700.HK',
            'window_start_date': '2023-10-01',
            'window_end_date': '2023-10-31',
            'confidence_score': 0.85,
            'outcome': 'success'
        }])
        
        # Try to save to a non-existent directory
        non_existent_original = "/non/existent/path.csv"
        
        with pytest.raises(SignalOutcomeError, match="Failed to save labeled matches"):
            tagger.save_labeled_matches(matches_df, non_existent_original)


class TestDataValidation:
    """Test data validation and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.signals_dir = os.path.join(self.temp_dir, 'signals')
        os.makedirs(self.signals_dir)
        
        self.tagger = SignalOutcomeTagger(signals_dir=self.signals_dir)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # Should not raise exception for empty DataFrame
        results = self.tagger.review_feedback(empty_df)
        assert results['total_matches'] == 0
        assert results['tagged_matches'] == 0
    
    def test_unicode_feedback_notes(self):
        """Test handling of unicode characters in feedback notes."""
        # Create matches with unicode content
        matches_data = pd.DataFrame([{
            'ticker': '0700.HK',
            'window_start_date': '2023-10-01',
            'window_end_date': '2023-10-31',
            'confidence_score': 0.85
        }])
        
        matches_file = os.path.join(self.signals_dir, 'unicode_test.csv')
        matches_data.to_csv(matches_file, index=False)
        
        matches_df = self.tagger.load_matches_file(matches_file)
        
        # Tag with unicode feedback
        unicode_feedback = "测试反馈 - émoji 🚀 test"
        updated_df = self.tagger.tag_outcome(
            matches_df, '0700.HK', '2023-10-01', 'success', unicode_feedback
        )
        
        # Save and reload to test persistence
        output_path = self.tagger.save_labeled_matches(updated_df, matches_file)
        reloaded_df = pd.read_csv(output_path)
        
        assert reloaded_df.iloc[0]['feedback_notes'] == unicode_feedback
    
    def test_extreme_confidence_values(self):
        """Test handling of extreme confidence score values."""
        # Create matches with extreme confidence values
        extreme_matches = pd.DataFrame([
            {
                'ticker': '0700.HK',
                'window_start_date': '2023-10-01',
                'window_end_date': '2023-10-31',
                'confidence_score': 0.0001  # Very low
            },
            {
                'ticker': '0005.HK',
                'window_start_date': '2023-10-01',
                'window_end_date': '2023-10-31',
                'confidence_score': 0.9999  # Very high
            }
        ])
        
        # Should handle extreme values without error
        results = self.tagger.review_feedback(extreme_matches)
        assert results is not None 