"""
Interactive Pattern Analysis Module

This module provides the core business logic for interactive pattern analysis,
extracted from the 06_interactive_demo.py notebook for better maintainability
and reusability.

Key Components:
- InteractivePatternAnalyzer: Main analysis class
- PatternAnalysisConfig: Configuration container
- PatternAnalysisResult: Results container
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import warnings
from dataclasses import dataclass

# Import core modules
from src.feature_extractor import FeatureExtractor
from src.pattern_scanner import PatternScanner, ScanningConfig
from src.data_fetcher import fetch_hk_stocks, get_all_cached_tickers


@dataclass
class PatternAnalysisConfig:
    """Configuration for interactive pattern analysis"""
    min_confidence: float = 0.7
    max_stocks_to_scan: Optional[int] = None
    show_progress: bool = True
    quiet_mode: bool = False  # Add quiet mode to reduce verbose output
    window_size: int = 30
    max_windows_per_ticker: int = 3
    context_days: int = 30
    min_pattern_days: int = 5
    model_type: str = "xgboost"


@dataclass
class PatternAnalysisResult:
    """Results container for pattern analysis"""
    matches_df: pd.DataFrame
    scanning_summary: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class SimplePatternConfig:
    """Lightweight config class for temporary models"""
    def __init__(self):
        self.model_type = "xgboost"
        self.training_approach = "interactive_demo"


class InteractivePatternAnalyzer:
    """
    Main class for interactive pattern analysis.
    
    Provides functionality to:
    1. Analyze user-defined positive patterns
    2. Train temporary models using negative examples
    3. Scan market for similar patterns
    4. Return ranked results with confidence scores
    """
    
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None):
        """
        Initialize the analyzer.
        
        Args:
            feature_extractor: Optional custom feature extractor
        """
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.temp_model_path = "temp_model.joblib"
        
    def analyze_pattern_similarity(self, 
                                 positive_ticker: str,
                                 start_date_str: str,
                                 end_date_str: str,
                                 negative_tickers_str: str,
                                 config: Optional[PatternAnalysisConfig] = None) -> PatternAnalysisResult:
        """
        Main analysis workflow to find similar patterns.
        
        Args:
            positive_ticker: Stock ticker for positive pattern example
            start_date_str: Pattern start date (YYYY-MM-DD)
            end_date_str: Pattern end date (YYYY-MM-DD)
            negative_tickers_str: Comma-separated negative example tickers
            config: Analysis configuration
            
        Returns:
            PatternAnalysisResult: Complete analysis results
        """
        if config is None:
            config = PatternAnalysisConfig()
            
        try:
            # Step 1: Validate inputs
            positive_example, negative_tickers = self._validate_inputs(
                positive_ticker, start_date_str, end_date_str, negative_tickers_str
            )
            
            if not config.quiet_mode:
                print(f"🔍 Analyzing positive pattern for {positive_ticker} from {start_date_str} to {end_date_str}...")
                print(f"📉 Using {len(negative_tickers)} negative examples: {', '.join(negative_tickers)}")
            
            # Step 2: Extract training features
            all_features, all_labels, feature_names = self._extract_training_features(
                positive_example, negative_tickers, config
            )
            
            # Step 3: Train temporary model
            model_package = self._train_temporary_model(all_features, all_labels, feature_names, config)
            
            # Step 4: Scan for similar patterns
            scan_results = self._scan_for_matches(model_package, positive_ticker, negative_tickers, config)
            
            # Step 5: Process and rank results
            final_results = self._process_results(scan_results, config)
            
            return PatternAnalysisResult(
                matches_df=final_results['matches_df'],
                scanning_summary=final_results['scanning_summary'],
                analysis_metadata={
                    'positive_ticker': positive_ticker,
                    'date_range': f"{start_date_str} to {end_date_str}",
                    'negative_examples': negative_tickers,
                    'training_samples': len(all_features),
                    'feature_count': len(feature_names),
                    'analysis_time': final_results.get('analysis_time', 0)
                },
                success=True
            )
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            return PatternAnalysisResult(
                matches_df=pd.DataFrame(),
                scanning_summary={},
                analysis_metadata={},
                success=False,
                error_message=error_msg
            )
        finally:
            # Clean up temporary model file
            if os.path.exists(self.temp_model_path):
                os.remove(self.temp_model_path)
    
    def _validate_inputs(self, positive_ticker: str, start_date_str: str, 
                        end_date_str: str, negative_tickers_str: str) -> Tuple[Dict, List[str]]:
        """
        Validate and parse input parameters.
        
        Args:
            positive_ticker: Stock ticker for positive example
            start_date_str: Start date string
            end_date_str: End date string
            negative_tickers_str: Comma-separated negative tickers
            
        Returns:
            Tuple of (positive_example_dict, negative_tickers_list)
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate dates
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        
        # Validate positive ticker and pattern length
        if not positive_ticker.endswith('.HK'):
            raise ValueError("Positive ticker must be a Hong Kong stock (.HK)")
        if (end_date - start_date).days < 5:
            raise ValueError("Pattern must be at least 5 days long")
            
        # Parse and validate negative tickers
        negative_tickers = [t.strip().upper() for t in negative_tickers_str.split(',') if t.strip()]
        if not negative_tickers:
            raise ValueError("Please provide at least one negative ticker")
            
        for ticker in negative_tickers:
            if not ticker.endswith('.HK'):
                raise ValueError(f"Invalid negative ticker: {ticker}. All tickers must end with .HK")
        
        positive_example = {
            'ticker': positive_ticker,
            'start_date': start_date,
            'end_date': end_date,
            'start_date_str': start_date_str,
            'end_date_str': end_date_str
        }
        
        return positive_example, negative_tickers
    
    def _extract_training_features(self, positive_example: Dict, negative_tickers: List[str], 
                                 config: PatternAnalysisConfig) -> Tuple[List[Dict], List[int], List[str]]:
        """
        Extract features for training from positive and negative examples.
        
        Args:
            positive_example: Positive pattern information
            negative_tickers: List of negative example tickers
            config: Analysis configuration
            
        Returns:
            Tuple of (all_features, all_labels, feature_names)
        """
        # Extract features for positive pattern
        context_start_date = positive_example['start_date'] - timedelta(days=config.context_days)
        
        # Suppress data fetching output in quiet mode
        import sys
        import os
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        devnull = open(os.devnull, 'w')
        
        try:
            if config.quiet_mode:
                sys.stdout = devnull
                sys.stderr = devnull
                
            data_dict = fetch_hk_stocks(
                [positive_example['ticker']], 
                str(context_start_date), 
                positive_example['end_date_str']
            )
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()
        
        if not data_dict or positive_example['ticker'] not in data_dict:
            raise ConnectionError(f"Could not fetch data for positive ticker {positive_example['ticker']}")
        
        full_data = data_dict[positive_example['ticker']]
        window_data = full_data.loc[positive_example['start_date_str']:positive_example['end_date_str']]
        prior_context_data = full_data.loc[:positive_example['start_date_str']].iloc[:-1]
        
        positive_features = self.feature_extractor.extract_features_from_window_data(
            window_data, prior_context_data, positive_example['ticker'], 
            positive_example['start_date_str'], positive_example['end_date_str'], full_data
        )
        
        if not positive_features:
            raise ValueError("Could not extract features from the positive pattern")
        
        # Initialize training data
        all_features = [positive_features]
        all_labels = [1]
        
        # Extract features for negative examples
        end_date = positive_example['end_date']
        
        # Suppress data fetching output for negative examples too
        devnull2 = open(os.devnull, 'w')
        try:
            if config.quiet_mode:
                sys.stdout = devnull2
                sys.stderr = devnull2
                
            negative_data = fetch_hk_stocks(
                negative_tickers, 
                (end_date - timedelta(days=365)).strftime('%Y-%m-%d'), 
                str(end_date)
            )
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull2.close()
        
        pattern_length = len(window_data)
        
        for neg_ticker, neg_df in negative_data.items():
            if len(neg_df) > pattern_length + config.context_days:
                # Random sampling for negative example
                rand_start = np.random.randint(0, len(neg_df) - pattern_length - config.context_days)
                neg_window = neg_df.iloc[rand_start + config.context_days : rand_start + config.context_days + pattern_length]
                neg_context = neg_df.iloc[rand_start : rand_start + config.context_days]
                
                neg_features = self.feature_extractor.extract_features_from_window_data(
                    neg_window, neg_context, neg_ticker, 
                    str(neg_window.index.min().date()), 
                    str(neg_window.index.max().date()), 
                    neg_df
                )
                
                if neg_features:
                    all_features.append(neg_features)
                    all_labels.append(0)
        
        if all_labels.count(0) == 0:
            raise ValueError("Failed to generate negative training samples from the provided tickers")
        
        # Get feature names (exclude metadata columns)
        full_features_df = pd.DataFrame(all_features)
        metadata_cols = ['ticker', 'start_date', 'end_date', 'label_type', 'notes']
        feature_names = [col for col in full_features_df.columns if col not in metadata_cols]
        
        return all_features, all_labels, feature_names
    
    def _train_temporary_model(self, all_features: List[Dict], all_labels: List[int], 
                             feature_names: List[str], config: PatternAnalysisConfig) -> Dict[str, Any]:
        """
        Train a temporary model for pattern detection.
        
        Args:
            all_features: List of feature dictionaries
            all_labels: List of labels (1 for positive, 0 for negative)
            feature_names: List of feature column names
            
        Returns:
            Model package dictionary
        """
        # Create clean training DataFrame
        full_features_df = pd.DataFrame(all_features)
        training_df_raw = full_features_df[feature_names]
        training_df = training_df_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Only show training info if not in quiet mode
        if not config.quiet_mode:
            print(f"🤖 Training model on {len(training_df)} samples "
                  f"({all_labels.count(1)} positive, {all_labels.count(0)} negative)...")
        
        try:
            import xgboost as xgb
            import joblib
            
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(training_df, pd.Series(all_labels))
            
            # Create model package
            model_package = {
                'model': model,
                'scaler': None,
                'feature_names': feature_names,
                'config': SimplePatternConfig(),
                'metadata': {
                    'training_date': datetime.now().isoformat(),
                    'n_samples': len(training_df),
                    'n_features': len(feature_names),
                    'class_distribution': pd.Series(all_labels).value_counts().to_dict()
                }
            }
            
            # Save model package
            joblib.dump(model_package, self.temp_model_path)
            return model_package
            
        except Exception as e:
            raise Exception(f"Model training failed: {e}")
    
    def _scan_for_matches(self, model_package: Dict[str, Any], positive_ticker: str, 
                         negative_tickers: List[str], config: PatternAnalysisConfig):
        """
        Scan market for similar patterns using trained model.
        
        Args:
            model_package: Trained model package
            positive_ticker: Original positive ticker (to exclude)
            negative_tickers: Negative tickers (to exclude)
            config: Analysis configuration
            
        Returns:
            Scanning results
        """
        # Completely suppress discovery output in quiet mode
        if not config.quiet_mode:
            print(f"🔎 Discovering available stocks from cached data...")
        
        all_available_stocks = get_all_cached_tickers()
        
        if not all_available_stocks:
            raise ValueError("No cached stock data found. Please run data collection first.")
        
        # Filter scan list (exclude positive and negative examples)
        scan_list = [t for t in all_available_stocks 
                    if t != positive_ticker and t not in negative_tickers]
        
        # Apply scan limit if specified
        if config.max_stocks_to_scan and config.max_stocks_to_scan < len(scan_list):
            scan_list = scan_list[:config.max_stocks_to_scan]
        
        # Only show summary info in quiet mode, no detailed discovery
        if not config.quiet_mode:
            print(f"📊 Found {len(all_available_stocks)} stocks with cached data")
            print(f"🔎 Scanning {len(scan_list)} stocks for similar patterns...")
        
        # Initialize scanner and run scan
        scanner = PatternScanner(model_path=self.temp_model_path)
        
        # Comprehensive output suppression
        import sys
        import os
        from io import StringIO
        
        # Store original stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        # Create devnull for complete suppression
        devnull = open(os.devnull, 'w')
        
        try:
            # Redirect all output to devnull in quiet mode
            if config.quiet_mode:
                sys.stdout = devnull
                sys.stderr = devnull
            else:
                # Capture but don't suppress in verbose mode
                sys.stdout = StringIO()
                sys.stderr = StringIO()
            
            # Scan with zero confidence to get all results (filter later)
            scan_results = scanner.scan_tickers(scan_list, ScanningConfig(
                min_confidence=0.0,  # Get all results
                max_windows_per_ticker=config.max_windows_per_ticker,
                save_results=False,
                top_matches_display=0  # Suppress internal display
            ))
            
            return scan_results
            
        finally:
            # Always restore original stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if 'devnull' in locals():
                devnull.close()
    
    def _process_results(self, scan_results, config: PatternAnalysisConfig) -> Dict[str, Any]:
        """
        Process and format scanning results.
        
        Args:
            scan_results: Raw scanning results
            config: Analysis configuration
            
        Returns:
            Processed results dictionary
        """
        if not scan_results or scan_results.matches_df.empty:
            return {
                'matches_df': pd.DataFrame(),
                'scanning_summary': {'matches_found': 0},
                'analysis_time': 0
            }
        
        # Get all results sorted by confidence
        all_results = scan_results.matches_df.sort_values('confidence_score', ascending=False)
        
        # Apply confidence threshold for "matches"
        matches_df = all_results[all_results['confidence_score'] >= config.min_confidence]
        
        # Use the config to determine verbosity level for result processing
        quiet_mode = getattr(config, 'quiet_mode', False)
        
        if not matches_df.empty:
            if not quiet_mode:
                print(f"\n✅ Found {len(matches_df)} patterns meeting {config.min_confidence:.0%} confidence threshold!")
                
                # Show confidence distribution
                high_conf = len(matches_df[matches_df['confidence_score'] >= 0.9])
                med_conf = len(matches_df[(matches_df['confidence_score'] >= 0.8) & (matches_df['confidence_score'] < 0.9)])
                low_conf = len(matches_df[matches_df['confidence_score'] < 0.8])
                
                print(f"📈 Confidence Distribution: {high_conf} high (≥90%), {med_conf} medium (80-90%), {low_conf} moderate (70-80%)")
                print(f"🎯 Top match: {matches_df.iloc[0]['ticker']} with {matches_df.iloc[0]['confidence_score']:.1%} confidence")
            
        else:
            # No matches meet threshold - show top candidates
            if not quiet_mode:
                print(f"\n⚠️  No patterns found meeting {config.min_confidence:.0%} confidence threshold")
                
                top_candidates = all_results.head(10)
                if not top_candidates.empty:
                    print(f"🎯 Best candidate: {top_candidates.iloc[0]['ticker']} with {top_candidates.iloc[0]['confidence_score']:.1%} confidence")
            
            # Return top candidates instead of empty results
            matches_df = all_results.head(10)
        
        return {
            'matches_df': matches_df,
            'scanning_summary': {
                'matches_found': len(matches_df),
                'total_candidates': len(all_results),
                'avg_confidence': all_results['confidence_score'].mean(),
                'max_confidence': all_results['confidence_score'].max()
            },
            'analysis_time': getattr(scan_results, 'scanning_time', 0)
        } 