"""
Pattern Scanner for Stock Trading Pattern Detection

This module provides the PatternScanner class that applies trained pattern detection 
models to scan multiple HK stocks using sliding windows to identify current pattern matches.
"""

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd

# Handle imports for both direct execution and package usage
try:
    from .feature_extractor import FeatureExtractor, FeatureExtractionError
    from .pattern_model_trainer import load_trained_model, ModelTrainingError
    from .data_fetcher import _load_cached_data
    from .hk_stock_universe import get_hk_stock_list_static, get_top_hk_stocks
except ImportError:
    from feature_extractor import FeatureExtractor, FeatureExtractionError
    from pattern_model_trainer import load_trained_model, ModelTrainingError
    from data_fetcher import _load_cached_data
    from hk_stock_universe import get_hk_stock_list_static, get_top_hk_stocks


# Configuration constants
SIGNALS_DIR = "signals"
DEFAULT_WINDOW_SIZE = 30
DEFAULT_MIN_CONFIDENCE = 0.70
DEFAULT_MAX_WINDOWS_PER_TICKER = 5
DEFAULT_TOP_MATCHES_DISPLAY = 5


class PatternScanningError(Exception):
    """Custom exception for pattern scanning errors."""
    pass


@dataclass
class ScanningConfig:
    """
    Configuration class for pattern scanning parameters.
    
    Attributes:
        window_size: Size of sliding window in days
        min_confidence: Minimum confidence threshold for matches
        max_windows_per_ticker: Maximum number of windows to evaluate per ticker
        top_matches_display: Number of top matches to display in console
        save_results: Whether to save results to file
        output_filename: Custom output filename (auto-generated if None)
        include_feature_values: Whether to include feature values in output
    """
    window_size: int = DEFAULT_WINDOW_SIZE
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    max_windows_per_ticker: int = DEFAULT_MAX_WINDOWS_PER_TICKER
    top_matches_display: int = DEFAULT_TOP_MATCHES_DISPLAY
    save_results: bool = True
    output_filename: Optional[str] = None
    include_feature_values: bool = False


@dataclass
class ScanningResults:
    """
    Results container for pattern scanning outcomes.
    
    Attributes:
        matches_df: DataFrame with all pattern matches
        scanning_summary: Summary statistics from scanning
        config: Scanning configuration used
        model_info: Information about the model used
        scanning_time: Time taken to complete scan
        output_path: Path to saved results file (if saved)
    """
    matches_df: pd.DataFrame
    scanning_summary: Dict[str, Any]
    config: ScanningConfig
    model_info: Dict[str, Any]
    scanning_time: float
    output_path: Optional[str] = None


class PatternScanner:
    """
    Main class for scanning stocks to detect trading patterns using trained models.
    
    This class provides functionality to:
    - Load and validate trained pattern detection models
    - Apply sliding window analysis across multiple tickers
    - Extract features using the same logic as training
    - Generate confidence-ranked pattern matches
    - Save timestamped results and display top matches
    """
    
    def __init__(self, 
                 model_path: str,
                 feature_extractor_config: Optional[Dict[str, Any]] = None,
                 signals_dir: str = SIGNALS_DIR):
        """
        Initialize PatternScanner.
        
        Args:
            model_path: Path to trained model file (.pkl)
            feature_extractor_config: Configuration for feature extraction
            signals_dir: Directory for saving output files
        """
        self.model_path = model_path
        self.signals_dir = signals_dir
        self.feature_extractor_config = feature_extractor_config or {}
        
        # Initialize components
        self.model_package = None
        self.feature_extractor = None
        
        # Create output directory
        self._ensure_signals_directory()
        
        # Load and validate model
        self._load_model_and_validate()
        
    def _ensure_signals_directory(self) -> None:
        """Create signals directory if it doesn't exist."""
        if not os.path.exists(self.signals_dir):
            os.makedirs(self.signals_dir)
            print(f"✓ Created signals directory: {self.signals_dir}/")
    
    def _load_model_and_validate(self) -> None:
        """Load trained model and validate compatibility."""
        try:
            print(f"🔄 Loading model from: {self.model_path}")
            self.model_package = load_trained_model(self.model_path)
            
            # Extract model components
            self.model = self.model_package['model']
            self.scaler = self.model_package.get('scaler')
            self.feature_names = self.model_package['feature_names']
            self.model_config = self.model_package['config']
            self.model_metadata = self.model_package['metadata']
            
            # Initialize feature extractor with matching configuration
            extractor_config = {
                'window_size': self.feature_extractor_config.get('window_size', DEFAULT_WINDOW_SIZE),
                'prior_context_days': self.feature_extractor_config.get('prior_context_days', 30),
                'support_lookback_days': self.feature_extractor_config.get('support_lookback_days', 10),
                'output_dir': "temp_features"  # Temporary directory for feature extraction
            }
            
            self.feature_extractor = FeatureExtractor(**extractor_config)
            
            print(f"✓ Model loaded successfully")
            print(f"  • Model type: {self.model_config.model_type}")
            print(f"  • Features expected: {len(self.feature_names)}")
            print(f"  • Training date: {self.model_metadata.get('training_date', 'Unknown')}")
            
        except Exception as e:
            raise PatternScanningError(f"Failed to load model: {e}")
    
    def _validate_feature_schema(self, extracted_features: Dict[str, Union[str, float]]) -> bool:
        """
        Validate that extracted features match model expectations.
        
        Args:
            extracted_features: Features extracted from a window
            
        Returns:
            bool: True if schema matches, False otherwise
        """
        try:
            # Get feature columns (exclude metadata columns)
            feature_columns = [col for col in extracted_features.keys() 
                             if col not in ['ticker', 'start_date', 'end_date', 'label_type', 'notes']]
            
            # Check if all expected features are present
            missing_features = set(self.feature_names) - set(feature_columns)
            extra_features = set(feature_columns) - set(self.feature_names)
            
            if missing_features:
                warnings.warn(f"Missing features: {missing_features}")
                return False
            
            if extra_features:
                warnings.warn(f"Extra features found: {extra_features}")
            
            return True
            
        except Exception as e:
            warnings.warn(f"Feature schema validation failed: {e}")
            return False
    
    def _extract_sliding_windows(self, 
                                ticker_data: pd.DataFrame, 
                                ticker: str,
                                config: ScanningConfig) -> List[Tuple[pd.DataFrame, pd.DataFrame, str, str]]:
        """
        Generate sliding windows from ticker data.
        
        Args:
            ticker_data: OHLCV data for the ticker
            ticker: Stock ticker symbol
            config: Scanning configuration
            
        Returns:
            List of tuples: (window_data, prior_context_data, start_date, end_date)
        """
        windows = []
        
        # Ensure we have enough data
        prior_context_days = getattr(self.feature_extractor, 'prior_context_days', 30)
        min_required_days = config.window_size + prior_context_days
        if len(ticker_data) < min_required_days:
            warnings.warn(f"Insufficient data for {ticker}: {len(ticker_data)} days < {min_required_days} required")
            return windows
        
        # Generate windows from most recent data backwards
        end_idx = len(ticker_data) - 1
        windows_generated = 0
        
        while windows_generated < config.max_windows_per_ticker and end_idx >= config.window_size - 1:
            # Define window boundaries
            start_idx = end_idx - config.window_size + 1
            
            # Extract window data
            window_data = ticker_data.iloc[start_idx:end_idx + 1].copy()
            
            # Extract prior context data
            prior_context_start = max(0, start_idx - prior_context_days)
            prior_context_end = start_idx - 1
            
            if prior_context_end >= prior_context_start:
                prior_context_data = ticker_data.iloc[prior_context_start:prior_context_end + 1].copy()
            else:
                prior_context_data = pd.DataFrame()
            
            # Get date strings
            start_date = window_data.index[0].strftime('%Y-%m-%d')
            end_date = window_data.index[-1].strftime('%Y-%m-%d')
            
            windows.append((window_data, prior_context_data, start_date, end_date))
            
            # Move to next window (step back by window_size for non-overlapping windows)
            end_idx -= config.window_size
            windows_generated += 1
        
        return windows
    
    def _extract_features_for_window(self, 
                                   window_data: pd.DataFrame,
                                   prior_context_data: pd.DataFrame,
                                   ticker: str,
                                   start_date: str,
                                   end_date: str,
                                   full_data: pd.DataFrame) -> Optional[Dict[str, Union[str, float]]]:
        """
        Extract features from a sliding window.
        
        Args:
            window_data: OHLCV data for the window
            prior_context_data: Prior context data
            ticker: Stock ticker
            start_date: Window start date
            end_date: Window end date
            full_data: Complete dataset for support level calculation
            
        Returns:
            Dict or None: Extracted features or None if extraction failed
        """
        try:
            # Check if feature extractor is available
            if self.feature_extractor is None:
                warnings.warn(f"Feature extractor not initialized for {ticker}")
                return None
            
            # Use the new method we added to FeatureExtractor
            features = self.feature_extractor.extract_features_from_window_data(
                window_data=window_data,
                prior_context_data=prior_context_data if not prior_context_data.empty else None,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                full_data=full_data
            )
            
            return features
            
        except Exception as e:
            warnings.warn(f"Feature extraction failed for {ticker} ({start_date} to {end_date}): {e}")
            return None
    
    def _predict_pattern_probability(self, features: Dict[str, Union[str, float]]) -> float:
        """
        Predict pattern probability using the trained model.
        
        Args:
            features: Extracted features dictionary
            
        Returns:
            float: Confidence score (probability of positive pattern)
        """
        try:
            # Prepare feature vector in correct order
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_vector.append(float(features[feature_name]))
                else:
                    warnings.warn(f"Missing feature: {feature_name}, using 0.0")
                    feature_vector.append(0.0)
            
            # Convert to numpy array and reshape for single prediction
            X = np.array(feature_vector).reshape(1, -1)
            
            # Apply scaling if scaler was used during training
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Get probability of positive class
            probabilities = self.model.predict_proba(X)
            positive_probability = probabilities[0][1]  # Probability of class 1 (positive)
            
            return float(positive_probability)
            
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            return 0.0
    
    def _filter_and_rank_results(self, 
                               results: List[Dict[str, Any]], 
                               config: ScanningConfig) -> pd.DataFrame:
        """
        Filter results by confidence and rank by score.
        
        Args:
            results: List of scanning results
            config: Scanning configuration
            
        Returns:
            pd.DataFrame: Filtered and ranked results
        """
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Filter by minimum confidence
        df = df[df['confidence_score'] >= config.min_confidence]
        
        # Sort by confidence score (descending)
        df = df.sort_values('confidence_score', ascending=False).reset_index(drop=True)
        
        # Add rank column
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _save_results(self, results_df: pd.DataFrame, config: ScanningConfig) -> str:
        """
        Save results to timestamped CSV file.
        
        Args:
            results_df: Results DataFrame
            config: Scanning configuration
            
        Returns:
            str: Path to saved file
        """
        if not config.save_results:
            return ""
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.output_filename:
            filename = config.output_filename
        else:
            filename = f"matches_{timestamp}.csv"
        
        output_path = os.path.join(self.signals_dir, filename)
        
        try:
            results_df.to_csv(output_path, index=False)
            print(f"💾 Results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            warnings.warn(f"Failed to save results: {e}")
            return ""
    
    def _display_top_matches(self, results_df: pd.DataFrame, config: ScanningConfig) -> None:
        """
        Display top matches in console.
        
        Args:
            results_df: Results DataFrame
            config: Scanning configuration
        """
        if results_df.empty:
            print("📊 No pattern matches found above confidence threshold")
            return
        
        print(f"\n🎯 Top {min(config.top_matches_display, len(results_df))} Pattern Matches:")
        print("=" * 80)
        
        for _, row in results_df.head(config.top_matches_display).iterrows():
            print(f"#{row['rank']:2d} | {row['ticker']:8s} | {row['window_start_date']} to {row['window_end_date']} | "
                  f"Confidence: {row['confidence_score']:.3f}")
    
    def _generate_scanning_summary(self, 
                                 tickers_scanned: List[str],
                                 results_df: pd.DataFrame,
                                 scanning_time: float) -> Dict[str, Any]:
        """
        Generate summary statistics from scanning results.
        
        Args:
            tickers_scanned: List of tickers that were scanned
            results_df: Results DataFrame
            scanning_time: Time taken for scanning
            
        Returns:
            Dict: Summary statistics
        """
        return {
            'total_tickers_scanned': len(tickers_scanned),
            'total_windows_evaluated': len(tickers_scanned) * DEFAULT_MAX_WINDOWS_PER_TICKER,  # Approximate
            'matches_found': len(results_df),
            'scanning_time_seconds': scanning_time,
            'average_confidence': results_df['confidence_score'].mean() if not results_df.empty else 0.0,
            'max_confidence': results_df['confidence_score'].max() if not results_df.empty else 0.0,
            'tickers_with_matches': results_df['ticker'].nunique() if not results_df.empty else 0
        }
    
    def scan_tickers(self, 
                    ticker_list: List[str], 
                    config: Optional[ScanningConfig] = None) -> ScanningResults:
        """
        Scan multiple tickers for pattern matches.
        
        Args:
            ticker_list: List of HK stock tickers to scan
            config: Scanning configuration (uses defaults if None)
            
        Returns:
            ScanningResults: Complete scanning results and metadata
        """
        config = config or ScanningConfig()
        start_time = datetime.now()
        
        print(f"🔍 Starting Pattern Scanning")
        print("=" * 50)
        print(f"  • Tickers to scan: {len(ticker_list)}")
        print(f"  • Window size: {config.window_size} days")
        print(f"  • Max windows per ticker: {config.max_windows_per_ticker}")
        print(f"  • Minimum confidence: {config.min_confidence}")
        print()
        
        all_results = []
        tickers_processed = []
        tickers_skipped = []
        
        for i, ticker in enumerate(ticker_list, 1):
            print(f"  Processing {i}/{len(ticker_list)}: {ticker}")
            
            try:
                # Load ticker data
                ticker_data = _load_cached_data(ticker)
                if ticker_data is None or ticker_data.empty:
                    print(f"    ⚠️  No data available - skipped")
                    tickers_skipped.append(ticker)
                    continue
                
                # Generate sliding windows
                windows = self._extract_sliding_windows(ticker_data, ticker, config)
                if not windows:
                    print(f"    ⚠️  Insufficient data for windows - skipped")
                    tickers_skipped.append(ticker)
                    continue
                
                tickers_processed.append(ticker)
                windows_evaluated = 0
                ticker_matches = 0
                
                # Process each window
                for window_data, prior_context_data, start_date, end_date in windows:
                    # Extract features
                    features = self._extract_features_for_window(
                        window_data, prior_context_data, ticker, start_date, end_date, ticker_data
                    )
                    
                    if features is None:
                        continue
                    
                    # Validate feature schema
                    if not self._validate_feature_schema(features):
                        warnings.warn(f"Feature schema mismatch for {ticker} window {start_date}-{end_date}")
                        continue
                    
                    # Predict pattern probability
                    confidence = self._predict_pattern_probability(features)
                    windows_evaluated += 1
                    
                    # Store result (will be filtered later)
                    result = {
                        'ticker': ticker,
                        'window_start_date': start_date,
                        'window_end_date': end_date,
                        'confidence_score': confidence
                    }
                    
                    # Add feature values if requested
                    if config.include_feature_values:
                        for feature_name in self.feature_names:
                            result[f'feature_{feature_name}'] = features.get(feature_name, 0.0)
                    
                    all_results.append(result)
                    
                    if confidence >= config.min_confidence:
                        ticker_matches += 1
                
                print(f"    ✓ {windows_evaluated} windows evaluated, {ticker_matches} matches found")
                
            except Exception as e:
                print(f"    ❌ Error processing {ticker}: {e}")
                tickers_skipped.append(ticker)
                continue
        
        # Filter and rank results
        results_df = self._filter_and_rank_results(all_results, config)
        
        # Calculate scanning time
        scanning_time = (datetime.now() - start_time).total_seconds()
        
        # Generate summary
        summary = self._generate_scanning_summary(tickers_processed, results_df, scanning_time)
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 Scanning Summary:")
        print(f"  • Total tickers scanned: {summary['total_tickers_scanned']}")
        print(f"  • Tickers skipped: {len(tickers_skipped)}")
        print(f"  • Pattern matches found: {summary['matches_found']}")
        print(f"  • Scanning time: {scanning_time:.2f} seconds")
        
        if tickers_skipped:
            print(f"  • Skipped tickers: {', '.join(tickers_skipped[:5])}" + 
                  (f" and {len(tickers_skipped)-5} more" if len(tickers_skipped) > 5 else ""))
        
        # Display top matches
        self._display_top_matches(results_df, config)
        
        # Save results
        output_path = self._save_results(results_df, config)
        
        # Prepare model info
        model_info = {
            'model_path': self.model_path,
            'model_type': self.model_config.model_type,
            'training_date': self.model_metadata.get('training_date'),
            'feature_count': len(self.feature_names)
        }
        
        return ScanningResults(
            matches_df=results_df,
            scanning_summary=summary,
            config=config,
            model_info=model_info,
            scanning_time=scanning_time,
            output_path=output_path
        )


def scan_hk_stocks_for_patterns(model_path: str,
                              ticker_list: Optional[List[str]] = None,
                              window_size: int = DEFAULT_WINDOW_SIZE,
                              min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                              max_windows_per_ticker: int = DEFAULT_MAX_WINDOWS_PER_TICKER,
                              **kwargs) -> ScanningResults:
    """
    Convenience function for scanning HK stocks for patterns.
    
    Args:
        model_path: Path to trained model file
        ticker_list: List of tickers to scan (uses top HK stocks if None)
        window_size: Size of sliding window in days
        min_confidence: Minimum confidence threshold
        max_windows_per_ticker: Maximum windows per ticker
        **kwargs: Additional configuration parameters
        
    Returns:
        ScanningResults: Complete scanning results
    """
    if ticker_list is None:
        ticker_list = get_top_hk_stocks(50)  # Default to top 50 HK stocks
    
    config = ScanningConfig(
        window_size=window_size,
        min_confidence=min_confidence,
        max_windows_per_ticker=max_windows_per_ticker,
        **kwargs
    )
    
    scanner = PatternScanner(model_path)
    return scanner.scan_tickers(ticker_list, config) 