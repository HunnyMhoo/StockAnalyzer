"""
Signal Outcome Tagger for Stock Trading Pattern Detection

This module provides the SignalOutcomeTagger class that enables manual tagging
of pattern match outcomes to track prediction accuracy and improve model training
through feedback loops.
"""

import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import shutil

import pandas as pd
import numpy as np

# Handle imports for both direct execution and package usage
try:
    from .pattern_visualizer import MatchVisualizationError
except ImportError:
    from pattern_visualizer import MatchVisualizationError


# Configuration constants
SIGNALS_DIR = "signals"
VALID_OUTCOMES = {'success', 'failure', 'uncertain'}
REQUIRED_MATCH_COLUMNS = ['ticker', 'window_start_date', 'window_end_date', 'confidence_score']
BACKUP_SUFFIX = "_backup"


class SignalOutcomeError(Exception):
    """Custom exception for signal outcome tagging errors."""
    pass


class SignalOutcomeTagger:
    """
    Main class for tagging signal outcomes and managing feedback data.
    
    This class provides functionality to:
    - Load and validate pattern match CSV files
    - Apply manual outcome tags (success/failure/uncertain)
    - Save labeled matches with feedback notes
    - Review and analyze feedback statistics
    - Support incremental tagging sessions
    """
    
    def __init__(self, signals_dir: str = SIGNALS_DIR, create_backups: bool = True):
        """
        Initialize SignalOutcomeTagger.
        
        Args:
            signals_dir: Directory containing match CSV files
            create_backups: Whether to create backup files before modifications
        """
        self.signals_dir = signals_dir
        self.create_backups = create_backups
        
        # Validate signals directory exists
        if not os.path.exists(self.signals_dir):
            raise SignalOutcomeError(f"Signals directory not found: {self.signals_dir}")
    
    def load_matches_file(self, match_file_path: str) -> pd.DataFrame:
        """
        Load and validate match CSV file.
        
        Args:
            match_file_path: Path to matches CSV file
            
        Returns:
            pd.DataFrame: Validated matches DataFrame
            
        Raises:
            SignalOutcomeError: If file cannot be loaded or validated
        """
        try:
            if not os.path.exists(match_file_path):
                raise SignalOutcomeError(f"Match file not found: {match_file_path}")
            
            # Load CSV file
            matches_df = pd.read_csv(match_file_path)
            
            if matches_df.empty:
                warnings.warn("Match file is empty")
                return matches_df
            
            # Validate required columns
            missing_columns = set(REQUIRED_MATCH_COLUMNS) - set(matches_df.columns)
            if missing_columns:
                raise SignalOutcomeError(
                    f"Missing required columns in match file: {missing_columns}"
                )
            
            # Add outcome columns if they don't exist yet
            if 'outcome' not in matches_df.columns:
                matches_df['outcome'] = pd.NA
            if 'feedback_notes' not in matches_df.columns:
                matches_df['feedback_notes'] = pd.NA
                
            # Add tagging metadata if not present
            if 'tagged_date' not in matches_df.columns:
                matches_df['tagged_date'] = pd.NA
            
            # Validate date formats
            for date_col in ['window_start_date', 'window_end_date']:
                try:
                    pd.to_datetime(matches_df[date_col])
                except ValueError as e:
                    raise SignalOutcomeError(f"Invalid date format in {date_col}: {e}")
            
            print(f"✓ Loaded {len(matches_df)} matches from {os.path.basename(match_file_path)}")
            return matches_df
            
        except Exception as e:
            if isinstance(e, SignalOutcomeError):
                raise
            raise SignalOutcomeError(f"Error loading match file: {e}")
    
    def validate_outcome(self, outcome: str) -> bool:
        """
        Validate that outcome value is acceptable.
        
        Args:
            outcome: Outcome value to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            SignalOutcomeError: If outcome is invalid
        """
        if pd.isna(outcome) or outcome is None:
            raise SignalOutcomeError("Outcome cannot be null or empty")
        
        outcome_str = str(outcome).lower().strip()
        
        if outcome_str not in VALID_OUTCOMES:
            raise SignalOutcomeError(
                f"Invalid outcome: '{outcome}'. Must be one of: {VALID_OUTCOMES}"
            )
        
        return True
    
    def find_match_by_key(self, 
                         matches_df: pd.DataFrame, 
                         ticker: str, 
                         window_start_date: str, 
                         window_end_date: Optional[str] = None) -> Tuple[int, pd.Series]:
        """
        Find a match by its composite key (ticker + dates).
        
        Args:
            matches_df: DataFrame containing matches
            ticker: Stock ticker symbol
            window_start_date: Window start date
            window_end_date: Window end date (optional for unique identification)
            
        Returns:
            Tuple[int, pd.Series]: Index and row of the matching record
            
        Raises:
            SignalOutcomeError: If match not found or multiple matches found
        """
        # Create filter conditions
        conditions = (matches_df['ticker'] == ticker) & \
                    (matches_df['window_start_date'] == window_start_date)
        
        if window_end_date is not None:
            conditions = conditions & (matches_df['window_end_date'] == window_end_date)
        
        # Find matching rows
        matching_rows = matches_df[conditions]
        
        if len(matching_rows) == 0:
            raise SignalOutcomeError(
                f"No match found for {ticker} starting {window_start_date}"
            )
        elif len(matching_rows) > 1:
            if window_end_date is None:
                # Multiple matches found, need end_date for disambiguation
                raise SignalOutcomeError(
                    f"Multiple matches found for {ticker} starting {window_start_date}. "
                    f"Please specify window_end_date for disambiguation."
                )
            else:
                raise SignalOutcomeError(
                    f"Multiple matches found for {ticker} ({window_start_date} to {window_end_date}). "
                    f"Data may have duplicates."
                )
        
        # Return index and row
        idx = matching_rows.index[0]
        return idx, matching_rows.iloc[0]
    
    def tag_outcome(self, 
                   matches_df: pd.DataFrame,
                   ticker: str,
                   window_start_date: str,
                   outcome: str,
                   feedback_notes: Optional[str] = None,
                   window_end_date: Optional[str] = None,
                   overwrite: bool = False) -> pd.DataFrame:
        """
        Tag a single match with outcome and feedback.
        
        Args:
            matches_df: DataFrame containing matches
            ticker: Stock ticker symbol
            window_start_date: Window start date
            outcome: Outcome value ('success', 'failure', 'uncertain')
            feedback_notes: Optional feedback notes
            window_end_date: Window end date for disambiguation
            overwrite: Whether to overwrite existing tags
            
        Returns:
            pd.DataFrame: Updated matches DataFrame
            
        Raises:
            SignalOutcomeError: If match not found or validation fails
        """
        # Validate outcome
        self.validate_outcome(outcome)
        
        # Find the match
        idx, existing_row = self.find_match_by_key(
            matches_df, ticker, window_start_date, window_end_date
        )
        
        # Check if already tagged
        if not pd.isna(existing_row['outcome']) and not overwrite:
            current_outcome = existing_row['outcome']
            raise SignalOutcomeError(
                f"Match for {ticker} ({window_start_date}) already tagged as '{current_outcome}'. "
                f"Use overwrite=True to replace existing tag."
            )
        
        # Apply the tag
        matches_df_updated = matches_df.copy()
        matches_df_updated.loc[idx, 'outcome'] = outcome.lower().strip()
        matches_df_updated.loc[idx, 'feedback_notes'] = feedback_notes or ""
        matches_df_updated.loc[idx, 'tagged_date'] = datetime.now().isoformat()
        
        # Confirmation message
        action = "Updated" if not pd.isna(existing_row['outcome']) else "Added"
        end_date_str = f" to {window_end_date}" if window_end_date else ""
        print(f"✓ {action} outcome tag for {ticker} ({window_start_date}{end_date_str}): {outcome}")
        
        return matches_df_updated
    
    def tag_batch_outcomes(self, 
                          matches_df: pd.DataFrame,
                          outcome_tags: List[Dict[str, Any]],
                          overwrite: bool = False) -> pd.DataFrame:
        """
        Apply multiple outcome tags in batch.
        
        Args:
            matches_df: DataFrame containing matches
            outcome_tags: List of tag dictionaries with keys:
                         'ticker', 'window_start_date', 'outcome', 
                         optional: 'window_end_date', 'feedback_notes'
            overwrite: Whether to overwrite existing tags
            
        Returns:
            pd.DataFrame: Updated matches DataFrame
        """
        updated_df = matches_df.copy()
        successful_tags = 0
        failed_tags = []
        
        for i, tag_info in enumerate(outcome_tags):
            try:
                # Extract required fields
                ticker = tag_info.get('ticker')
                window_start_date = tag_info.get('window_start_date')
                outcome = tag_info.get('outcome')
                
                # Validate required fields
                if not all([ticker, window_start_date, outcome]):
                    raise SignalOutcomeError(
                        f"Missing required fields in tag {i+1}: ticker, window_start_date, outcome"
                    )
                
                # Extract optional fields
                window_end_date = tag_info.get('window_end_date')
                feedback_notes = tag_info.get('feedback_notes')
                
                # Apply the tag
                updated_df = self.tag_outcome(
                    updated_df, ticker, window_start_date, outcome,
                    feedback_notes, window_end_date, overwrite
                )
                successful_tags += 1
                
            except SignalOutcomeError as e:
                failed_tags.append({
                    'index': i + 1,
                    'tag_info': tag_info,
                    'error': str(e)
                })
                warnings.warn(f"Failed to apply tag {i+1}: {e}")
        
        # Summary
        print(f"\n📊 Batch tagging summary:")
        print(f"   ✓ Successfully tagged: {successful_tags}")
        print(f"   ❌ Failed tags: {len(failed_tags)}")
        
        if failed_tags:
            print(f"\n⚠️  Failed tag details:")
            for failed in failed_tags:
                print(f"   Tag {failed['index']}: {failed['error']}")
        
        return updated_df
    
    def save_labeled_matches(self, 
                           matches_df: pd.DataFrame, 
                           original_file_path: str,
                           output_file_path: Optional[str] = None) -> str:
        """
        Save labeled matches to file with backup support.
        
        Args:
            matches_df: DataFrame with labeled matches
            original_file_path: Path to original match file
            output_file_path: Optional custom output path
            
        Returns:
            str: Path to saved labeled file
        """
        try:
            # Generate output file path if not provided
            if output_file_path is None:
                base_name = os.path.basename(original_file_path)
                name_without_ext = os.path.splitext(base_name)[0]
                output_file_path = os.path.join(
                    self.signals_dir, 
                    f"{name_without_ext}_labeled.csv"
                )
            
            # Create backup of original file if it's the first time tagging
            if self.create_backups and not output_file_path.endswith('_labeled.csv'):
                backup_path = original_file_path + BACKUP_SUFFIX
                if not os.path.exists(backup_path):
                    shutil.copy2(original_file_path, backup_path)
                    print(f"📁 Created backup: {os.path.basename(backup_path)}")
            
            # Save labeled matches
            matches_df.to_csv(output_file_path, index=False)
            print(f"💾 Labeled matches saved to: {os.path.basename(output_file_path)}")
            
            return output_file_path
            
        except Exception as e:
            raise SignalOutcomeError(f"Failed to save labeled matches: {e}")
    
    def review_feedback(self, 
                       matches_df: pd.DataFrame,
                       confidence_bands: List[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Analyze feedback statistics by confidence bands and outcomes.
        
        Args:
            matches_df: DataFrame with labeled matches
            confidence_bands: List of (min, max) confidence ranges
            
        Returns:
            Dict containing feedback analysis results
        """
        if confidence_bands is None:
            confidence_bands = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.0, 0.6)]
        
        # Filter to only tagged matches
        tagged_matches = matches_df[matches_df['outcome'].notna()].copy()
        
        if tagged_matches.empty:
            print("⚠️  No tagged matches found for feedback analysis")
            return {
                'total_matches': len(matches_df),
                'tagged_matches': 0,
                'tagging_rate': 0.0,
                'confidence_bands': {},
                'outcome_summary': {}
            }
        
        # Overall statistics
        total_matches = len(matches_df)
        tagged_count = len(tagged_matches)
        tagging_rate = tagged_count / total_matches if total_matches > 0 else 0
        
        # Outcome summary
        outcome_counts = tagged_matches['outcome'].value_counts()
        outcome_percentages = tagged_matches['outcome'].value_counts(normalize=True) * 100
        
        # Confidence band analysis
        confidence_analysis = {}
        
        for min_conf, max_conf in confidence_bands:
            band_name = f"{min_conf:.1f}-{max_conf:.1f}"
            
            # Filter matches in this confidence band
            band_matches = tagged_matches[
                (tagged_matches['confidence_score'] >= min_conf) & 
                (tagged_matches['confidence_score'] < max_conf)
            ]
            
            if len(band_matches) == 0:
                confidence_analysis[band_name] = {
                    'count': 0,
                    'outcomes': {},
                    'success_rate': None
                }
                continue
            
            # Calculate outcomes in this band
            band_outcomes = band_matches['outcome'].value_counts().to_dict()
            
            # Calculate success rate
            success_count = band_outcomes.get('success', 0)
            total_decisive = success_count + band_outcomes.get('failure', 0)
            success_rate = success_count / total_decisive if total_decisive > 0 else None
            
            confidence_analysis[band_name] = {
                'count': len(band_matches),
                'outcomes': band_outcomes,
                'success_rate': success_rate,
                'avg_confidence': band_matches['confidence_score'].mean()
            }
        
        # Compile results
        results = {
            'total_matches': total_matches,
            'tagged_matches': tagged_count,
            'tagging_rate': tagging_rate,
            'outcome_summary': {
                'counts': outcome_counts.to_dict(),
                'percentages': outcome_percentages.to_dict()
            },
            'confidence_bands': confidence_analysis,
            'analysis_date': datetime.now().isoformat()
        }
        
        # Display summary
        self._display_feedback_summary(results)
        
        return results
    
    def _display_feedback_summary(self, results: Dict[str, Any]) -> None:
        """Display formatted feedback analysis summary."""
        print("📊 Feedback Analysis Summary")
        print("=" * 50)
        
        # Overall statistics
        print(f"📈 Overall Statistics:")
        print(f"   Total matches: {results['total_matches']}")
        print(f"   Tagged matches: {results['tagged_matches']}")
        print(f"   Tagging rate: {results['tagging_rate']:.1%}")
        
        # Outcome breakdown
        if results['outcome_summary']['counts']:
            print(f"\n🎯 Outcome Breakdown:")
            for outcome, count in results['outcome_summary']['counts'].items():
                percentage = results['outcome_summary']['percentages'][outcome]
                print(f"   {outcome.title()}: {count} ({percentage:.1f}%)")
        
        # Confidence band analysis
        print(f"\n📊 Performance by Confidence Band:")
        print(f"{'Band':<12} {'Count':<8} {'Success':<10} {'Failure':<10} {'Success%':<10}")
        print("-" * 60)
        
        for band_name, analysis in results['confidence_bands'].items():
            if analysis['count'] == 0:
                continue
                
            success = analysis['outcomes'].get('success', 0)
            failure = analysis['outcomes'].get('failure', 0)
            success_rate = analysis['success_rate']
            success_pct = f"{success_rate:.1%}" if success_rate is not None else "N/A"
            
            print(f"{band_name:<12} {analysis['count']:<8} {success:<10} {failure:<10} {success_pct:<10}")
    
    def find_available_match_files(self) -> List[str]:
        """
        Find all available match CSV files in the signals directory.
        
        Returns:
            List[str]: List of match file paths
        """
        match_files = []
        
        if not os.path.exists(self.signals_dir):
            return match_files
        
        for filename in os.listdir(self.signals_dir):
            if filename.startswith('matches_') and filename.endswith('.csv'):
                # Skip already labeled files
                if not filename.endswith('_labeled.csv'):
                    match_files.append(os.path.join(self.signals_dir, filename))
        
        return sorted(match_files)
    
    def get_match_summary(self, matches_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for a matches DataFrame.
        
        Args:
            matches_df: DataFrame containing matches
            
        Returns:
            Dict containing summary statistics
        """
        if matches_df.empty:
            return {
                'total_matches': 0,
                'tagged_count': 0,
                'untagged_count': 0,
                'confidence_range': (None, None),
                'date_range': (None, None)
            }
        
        total_matches = len(matches_df)
        
        # Count tagged vs untagged
        has_outcome_col = 'outcome' in matches_df.columns
        if has_outcome_col:
            tagged_count = (~matches_df['outcome'].isna()).sum()
            untagged_count = total_matches - tagged_count
        else:
            tagged_count = 0
            untagged_count = total_matches
        
        # Confidence range
        conf_min = matches_df['confidence_score'].min()
        conf_max = matches_df['confidence_score'].max()
        
        # Date range
        start_dates = pd.to_datetime(matches_df['window_start_date'])
        date_min = start_dates.min().strftime('%Y-%m-%d') if not start_dates.empty else None
        date_max = start_dates.max().strftime('%Y-%m-%d') if not start_dates.empty else None
        
        return {
            'total_matches': total_matches,
            'tagged_count': tagged_count,
            'untagged_count': untagged_count,
            'confidence_range': (conf_min, conf_max),
            'date_range': (date_min, date_max),
            'tickers': sorted(matches_df['ticker'].unique().tolist())
        }


# Convenience functions for easy notebook usage
def load_latest_matches(signals_dir: str = SIGNALS_DIR) -> Tuple[str, pd.DataFrame]:
    """
    Load the most recent match file.
    
    Args:
        signals_dir: Directory containing match files
        
    Returns:
        Tuple[str, pd.DataFrame]: File path and loaded DataFrame
    """
    tagger = SignalOutcomeTagger(signals_dir)
    match_files = tagger.find_available_match_files()
    
    if not match_files:
        raise SignalOutcomeError("No match files found in signals directory")
    
    latest_file = match_files[-1]  # Files are sorted, so last is most recent
    matches_df = tagger.load_matches_file(latest_file)
    
    return latest_file, matches_df


def quick_tag_outcome(ticker: str, 
                     window_start_date: str, 
                     outcome: str,
                     feedback_notes: str = None,
                     signals_dir: str = SIGNALS_DIR) -> None:
    """
    Quick function to tag a single outcome and save immediately.
    
    Args:
        ticker: Stock ticker symbol
        window_start_date: Window start date
        outcome: Outcome value
        feedback_notes: Optional feedback notes
        signals_dir: Directory containing match files
    """
    # Load latest matches
    file_path, matches_df = load_latest_matches(signals_dir)
    
    # Apply tag
    tagger = SignalOutcomeTagger(signals_dir)
    updated_df = tagger.tag_outcome(
        matches_df, ticker, window_start_date, outcome, feedback_notes
    )
    
    # Save immediately
    tagger.save_labeled_matches(updated_df, file_path)
    print("✅ Quick tag applied and saved!")


def review_latest_feedback(signals_dir: str = SIGNALS_DIR) -> Dict[str, Any]:
    """
    Quick function to review feedback for the latest labeled matches.
    
    Args:
        signals_dir: Directory containing match files
        
    Returns:
        Dict containing feedback analysis
    """
    # Look for labeled files first
    labeled_files = []
    if os.path.exists(signals_dir):
        for filename in os.listdir(signals_dir):
            if filename.endswith('_labeled.csv'):
                labeled_files.append(os.path.join(signals_dir, filename))
    
    if not labeled_files:
        print("⚠️  No labeled match files found")
        return {}
    
    # Load most recent labeled file
    latest_labeled = sorted(labeled_files)[-1]
    
    tagger = SignalOutcomeTagger(signals_dir)
    matches_df = tagger.load_matches_file(latest_labeled)
    
    return tagger.review_feedback(matches_df) 