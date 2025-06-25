"""
Feature Extractor for Stock Trading Patterns

This module provides the main FeatureExtractor class that converts labeled stock
pattern windows into structured numerical features for machine learning.
"""

import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Handle imports for both direct execution and package usage
try:
    from ..patterns.labeler import PatternLabel, load_labeled_patterns
    from ..data.fetcher import fetch_hk_stocks, _get_cache_filename, _load_cached_data
    from .indicators import (
        simple_moving_average,
        relative_strength_index,
        macd,
        price_volatility,
        volume_average_ratio,
        find_recent_support_level,
        calculate_linear_trend_slope,
        detect_false_support_break,
        calculate_drawdown_metrics,
        calculate_candle_patterns,
    )
except ImportError:
    from ..patterns.labeler import PatternLabel, load_labeled_patterns
from ..data.fetcher import fetch_hk_stocks, _get_cache_filename, _load_cached_data
from .indicators import (
        simple_moving_average,
        relative_strength_index,
        macd,
        price_volatility,
        volume_average_ratio,
        find_recent_support_level,
        calculate_linear_trend_slope,
        detect_false_support_break,
        calculate_drawdown_metrics,
        calculate_candle_patterns,
    )


# Configuration constants
FEATURES_DIR = "features"
DEFAULT_OUTPUT_FILE = "labeled_features.csv"
DEFAULT_WINDOW_SIZE = 30
DEFAULT_PRIOR_CONTEXT_DAYS = 30
DEFAULT_SUPPORT_LOOKBACK_DAYS = 10


class FeatureExtractionError(Exception):
    """Custom exception for feature extraction errors."""
    pass


@dataclass
class FeatureWindow:
    """
    Data class representing a feature extraction window.
    
    Attributes:
        ticker: Stock ticker
        start_date: Window start date
        end_date: Window end date
        label_type: Pattern label type
        notes: Pattern notes
        window_data: OHLCV data for the window period
        prior_context_data: OHLCV data for prior context period
    """
    ticker: str
    start_date: str
    end_date: str
    label_type: str
    notes: str
    window_data: pd.DataFrame
    prior_context_data: Optional[pd.DataFrame] = None


class FeatureExtractor:
    """
    Main class for extracting numerical features from labeled stock patterns.
    """
    
    def __init__(self, 
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 prior_context_days: int = DEFAULT_PRIOR_CONTEXT_DAYS,
                 support_lookback_days: int = DEFAULT_SUPPORT_LOOKBACK_DAYS,
                 output_dir: str = FEATURES_DIR):
        """
        Initialize FeatureExtractor.
        
        Args:
            window_size: Size of pattern window in days
            prior_context_days: Days of prior context for trend analysis
            support_lookback_days: Days to look back for support level detection
            output_dir: Directory for output files
        """
        self.window_size = window_size
        self.prior_context_days = prior_context_days
        self.support_lookback_days = support_lookback_days
        self.output_dir = output_dir
        self._ensure_output_directory()
    
    def _ensure_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✓ Created features directory: {self.output_dir}/")
    
    def _load_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for a ticker from cache.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            DataFrame or None: OHLCV data if available
        """
        try:
            data = _load_cached_data(ticker)
            if data is None or data.empty:
                warnings.warn(f"No cached data found for {ticker}")
                return None
            return data
        except Exception as e:
            warnings.warn(f"Failed to load data for {ticker}: {e}")
            return None
    
    def _extract_window_data(self, 
                           full_data: pd.DataFrame, 
                           start_date: str, 
                           end_date: str,
                           include_prior_context: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Extract window data and prior context from full dataset.
        
        Args:
            full_data: Complete OHLCV dataset
            start_date: Window start date
            end_date: Window end date
            include_prior_context: Whether to include prior context data
            
        Returns:
            Tuple of (window_data, prior_context_data)
        """
        try:
            # Convert dates to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Extract window data
            window_mask = (full_data.index >= start_dt) & (full_data.index <= end_dt)
            window_data = full_data.loc[window_mask].copy()
            
            prior_context_data = None
            if include_prior_context:
                # Calculate prior context period
                prior_start = start_dt - timedelta(days=self.prior_context_days)
                prior_end = start_dt - timedelta(days=1)
                
                # Extract prior context data
                prior_mask = (full_data.index >= prior_start) & (full_data.index <= prior_end)
                prior_context_data = full_data.loc[prior_mask].copy()
                
                if prior_context_data.empty:
                    prior_context_data = None
            
            return window_data, prior_context_data
            
        except Exception as e:
            raise FeatureExtractionError(f"Failed to extract window data: {e}")
    
    def _validate_window_data(self, window_data: pd.DataFrame, ticker: str) -> bool:
        """
        Validate that window data is sufficient for feature extraction.
        
        Args:
            window_data: Window OHLCV data
            ticker: Stock ticker (for error messages)
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if window_data.empty:
            warnings.warn(f"Empty window data for {ticker}")
            return False
        
        if len(window_data) < 5:  # Minimum required days
            warnings.warn(f"Insufficient window data for {ticker}: {len(window_data)} days")
            return False
        
        # Check for required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = set(required_cols) - set(window_data.columns)
        if missing_cols:
            warnings.warn(f"Missing columns for {ticker}: {missing_cols}")
            return False
        
        # Check for all-null values
        if window_data[required_cols].isnull().all().any():
            warnings.warn(f"All-null values found in {ticker} data")
            return False
        
        return True
    
    def _calculate_trend_features(self, 
                                window_data: pd.DataFrame, 
                                prior_context_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate trend context features.
        
        Args:
            window_data: Pattern window data
            prior_context_data: Prior context data (30 days before window)
            
        Returns:
            Dict: Trend features
        """
        features = {}
        
        if prior_context_data is not None and len(prior_context_data) >= 5:
            # Prior trend return
            prior_start_price = prior_context_data["Close"].iloc[0]
            prior_end_price = prior_context_data["Close"].iloc[-1]
            features["prior_trend_return"] = float(((prior_end_price - prior_start_price) / prior_start_price) * 100)
            
            # Above SMA 50 ratio (using available data if less than 50 days)
            sma_window = min(50, len(prior_context_data))
            sma_50 = simple_moving_average(prior_context_data["Close"], sma_window)
            if not sma_50.empty:
                above_sma_days = (prior_context_data["Close"] > sma_50).sum()
                features["above_sma_50_ratio"] = float((above_sma_days / len(prior_context_data)) * 100)
            else:
                features["above_sma_50_ratio"] = 0.0
            
            # Trend angle (linear regression slope)
            features["trend_angle"] = calculate_linear_trend_slope(prior_context_data["Close"])
        else:
            # Default values when prior context is unavailable
            features["prior_trend_return"] = 0.0
            features["above_sma_50_ratio"] = 0.0
            features["trend_angle"] = 0.0
        
        return features
    
    def _calculate_correction_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate correction phase features.
        
        Args:
            window_data: Pattern window data
            
        Returns:
            Dict: Correction features
        """
        features = {}
        
        # Drawdown and recovery metrics
        drawdown_pct, recovery_return_pct = calculate_drawdown_metrics(window_data["Close"])
        features["drawdown_pct"] = drawdown_pct
        features["recovery_return_pct"] = recovery_return_pct
        
        # Down day ratio (first half of window)
        first_half_length = len(window_data) // 2
        if first_half_length > 0:
            first_half_data = window_data.iloc[:first_half_length]
            red_candles = calculate_candle_patterns(
                first_half_data["Open"],
                first_half_data["High"],
                first_half_data["Low"],
                first_half_data["Close"]
            )
            features["down_day_ratio"] = float((red_candles.sum() / len(red_candles)) * 100)
        else:
            features["down_day_ratio"] = 0.0
        
        return features
    
    def _calculate_support_break_features(self, 
                                        window_data: pd.DataFrame, 
                                        full_data: pd.DataFrame,
                                        window_start_idx: int) -> Dict[str, float]:
        """
        Calculate false support break features.
        
        Args:
            window_data: Pattern window data
            full_data: Complete dataset for support level calculation
            window_start_idx: Index of window start in full dataset
            
        Returns:
            Dict: Support break features
        """
        features = {}
        
        # Find support level from prior period
        support_level = find_recent_support_level(
            full_data["Low"], 
            window_start_idx, 
            self.support_lookback_days
        )
        
        if support_level is not None:
            features["support_level"] = float(support_level)
            
            # Calculate support break depth
            window_low = window_data["Low"].min()
            if window_low < support_level:
                features["support_break_depth_pct"] = float(((support_level - window_low) / support_level) * 100)
            else:
                features["support_break_depth_pct"] = 0.0
            
            # Detect false break
            is_false_break, recovery_days = detect_false_support_break(
                window_data["Close"], 
                support_level
            )
            features["false_break_flag"] = float(1.0 if is_false_break else 0.0)
            features["recovery_days"] = float(recovery_days if recovery_days >= 0 else 0)
            
            # Recovery volume ratio
            if is_false_break and recovery_days >= 0 and recovery_days < len(window_data):
                recovery_volume = window_data["Volume"].iloc[recovery_days]
                avg_volume_20 = volume_average_ratio(window_data["Volume"], 20)
                if not avg_volume_20.empty and recovery_days < len(avg_volume_20):
                    volume_ratio = recovery_volume / window_data["Volume"].rolling(20).mean().iloc[recovery_days]
                    features["recovery_volume_ratio"] = float(volume_ratio) if not np.isnan(volume_ratio) else 1.0
                else:
                    features["recovery_volume_ratio"] = 1.0
            else:
                features["recovery_volume_ratio"] = 1.0
        else:
            # Default values when support level cannot be determined
            features["support_level"] = float(window_data["Low"].min())
            features["support_break_depth_pct"] = 0.0
            features["false_break_flag"] = 0.0
            features["recovery_days"] = 0.0
            features["recovery_volume_ratio"] = 1.0
        
        return features
    
    def _calculate_technical_indicators(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate technical indicator features.
        
        Args:
            window_data: Pattern window data
            
        Returns:
            Dict: Technical indicator features
        """
        features = {}
        
        # Final day close price for calculations
        final_close = window_data["Close"].iloc[-1]
        
        # Moving averages (final day values)
        sma_5 = simple_moving_average(window_data["Close"], 5)
        sma_10 = simple_moving_average(window_data["Close"], 10)
        sma_20 = simple_moving_average(window_data["Close"], 20)
        
        features["sma_5"] = float(sma_5.iloc[-1]) if not sma_5.empty and not pd.isna(sma_5.iloc[-1]) else final_close
        features["sma_10"] = float(sma_10.iloc[-1]) if not sma_10.empty and not pd.isna(sma_10.iloc[-1]) else final_close
        features["sma_20"] = float(sma_20.iloc[-1]) if not sma_20.empty and not pd.isna(sma_20.iloc[-1]) else final_close
        
        # RSI (final day value)
        rsi_14 = relative_strength_index(window_data["Close"], 14)
        features["rsi_14"] = float(rsi_14.iloc[-1]) if not rsi_14.empty and not pd.isna(rsi_14.iloc[-1]) else 50.0
        
        # MACD difference (final day value)
        macd_line, signal_line, histogram = macd(window_data["Close"])
        features["macd_diff"] = float(histogram.iloc[-1]) if not histogram.empty and not pd.isna(histogram.iloc[-1]) else 0.0
        
        # Volatility
        volatility = price_volatility(window_data["Close"], min(20, len(window_data)))
        features["volatility"] = float(volatility.iloc[-1]) if not volatility.empty and not pd.isna(volatility.iloc[-1]) else 0.0
        
        # Volume average ratio (final day)
        vol_ratio = volume_average_ratio(window_data["Volume"], min(20, len(window_data)))
        features["volume_avg_ratio"] = float(vol_ratio.iloc[-1]) if not vol_ratio.empty and not pd.isna(vol_ratio.iloc[-1]) else 1.0
        
        return features
    
    def extract_features_from_window_data(self, 
                                         window_data: pd.DataFrame,
                                         prior_context_data: Optional[pd.DataFrame],
                                         ticker: str,
                                         start_date: str,
                                         end_date: str,
                                         full_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Union[str, float]]]:
        """
        Extract features from window data without requiring a PatternLabel.
        
        This method is used by the PatternScanner for unlabeled feature extraction.
        
        Args:
            window_data: OHLCV data for the pattern window
            prior_context_data: Prior context data (can be None)
            ticker: Stock ticker symbol
            start_date: Window start date string
            end_date: Window end date string
            full_data: Complete dataset for support level calculation (optional)
            
        Returns:
            Dict or None: Feature dictionary or None if extraction failed
        """
        try:
            # Validate window data
            if not self._validate_window_data(window_data, ticker):
                return None
            
            # Find window start index in full dataset (for support break features)
            window_start_idx = 0
            if full_data is not None:
                try:
                    start_dt = pd.to_datetime(start_date)
                    matching_dates = full_data.index[full_data.index >= start_dt]
                    if len(matching_dates) > 0:
                        loc_result = full_data.index.get_loc(matching_dates[0])
                        window_start_idx = int(loc_result) if isinstance(loc_result, (int, np.integer)) else 0
                except Exception:
                    window_start_idx = 0
            
            # Initialize feature dictionary with metadata
            features: Dict[str, Union[str, float]] = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "label_type": "unlabeled",  # Mark as unlabeled
                "notes": "Pattern scanning extraction"
            }
            
            # Calculate feature categories
            trend_features = self._calculate_trend_features(window_data, prior_context_data)
            correction_features = self._calculate_correction_features(window_data)
            
            # For support break features, use full_data if available, otherwise use window_data
            support_data = full_data if full_data is not None else window_data
            support_features = self._calculate_support_break_features(window_data, support_data, window_start_idx)
            
            technical_features = self._calculate_technical_indicators(window_data)
            
            # Combine all features
            for key, value in trend_features.items():
                features[key] = value
            for key, value in correction_features.items():
                features[key] = value
            for key, value in support_features.items():
                features[key] = value
            for key, value in technical_features.items():
                features[key] = value
            
            return features
            
        except Exception as e:
            warnings.warn(f"Failed to extract features for {ticker} ({start_date} to {end_date}): {e}")
            return None

    def extract_features_from_label(self, label: PatternLabel) -> Optional[Dict[str, Union[str, float]]]:
        """
        Extract features from a single pattern label.
        
        Args:
            label: Pattern label to process
            
        Returns:
            Dict or None: Feature dictionary or None if extraction failed
        """
        try:
            # Load ticker data
            full_data = self._load_ticker_data(label.ticker)
            if full_data is None:
                return None
            
            # Extract window and prior context data
            window_data, prior_context_data = self._extract_window_data(
                full_data, label.start_date, label.end_date
            )
            
            # Validate window data
            if not self._validate_window_data(window_data, label.ticker):
                return None
            
            # Find window start index in full dataset
            start_dt = pd.to_datetime(label.start_date)
            loc_result = full_data.index.get_loc(full_data.index[full_data.index >= start_dt][0])
            window_start_idx = int(loc_result) if isinstance(loc_result, (int, np.integer)) else 0
            
            # Initialize feature dictionary with metadata
            features: Dict[str, Union[str, float]] = {
                "ticker": label.ticker,
                "start_date": label.start_date,
                "end_date": label.end_date,
                "label_type": label.label_type,
                "notes": label.notes
            }
            
            # Calculate feature categories
            trend_features = self._calculate_trend_features(window_data, prior_context_data)
            correction_features = self._calculate_correction_features(window_data)
            support_features = self._calculate_support_break_features(window_data, full_data, window_start_idx)
            technical_features = self._calculate_technical_indicators(window_data)
            
            # Combine all features
            for key, value in trend_features.items():
                features[key] = value
            for key, value in correction_features.items():
                features[key] = value
            for key, value in support_features.items():
                features[key] = value
            for key, value in technical_features.items():
                features[key] = value
            
            return features
            
        except Exception as e:
            warnings.warn(f"Failed to extract features for {label.ticker} ({label.start_date} to {label.end_date}): {e}")
            return None
    
    def extract_features_batch(self, 
                             labels: List[PatternLabel], 
                             save_to_file: bool = True,
                             output_filename: Optional[str] = None) -> pd.DataFrame:
        """
        Extract features from a batch of pattern labels.
        
        Args:
            labels: List of pattern labels to process
            save_to_file: Whether to save results to CSV file
            output_filename: Custom output filename (optional)
            
        Returns:
            pd.DataFrame: Features dataframe
        """
        print(f"🔄 Extracting features from {len(labels)} labeled patterns...")
        
        feature_list = []
        success_count = 0
        
        for i, label in enumerate(labels, 1):
            print(f"  Processing {i}/{len(labels)}: {label.ticker} ({label.start_date} to {label.end_date})")
            
            features = self.extract_features_from_label(label)
            if features is not None:
                feature_list.append(features)
                success_count += 1
            else:
                print(f"    ⚠️  Skipped due to insufficient data")
        
        # Create DataFrame
        if feature_list:
            df = pd.DataFrame(feature_list)
            print(f"✓ Successfully extracted features from {success_count}/{len(labels)} patterns")
            print(f"  Features shape: {df.shape}")
            
            # Save to file if requested
            if save_to_file:
                filename = output_filename or DEFAULT_OUTPUT_FILE
                output_path = os.path.join(self.output_dir, filename)
                df.to_csv(output_path, index=False)
                print(f"💾 Saved features to: {output_path}")
            
            return df
        else:
            print("❌ No features extracted - all patterns were skipped")
            return pd.DataFrame()
    
    def extract_features_from_file(self, 
                                 labels_file: Optional[str] = None,
                                 save_to_file: bool = True,
                                 output_filename: Optional[str] = None) -> pd.DataFrame:
        """
        Extract features from labeled patterns file.
        
        Args:
            labels_file: Path to labeled patterns JSON file (optional)
            save_to_file: Whether to save results to CSV file
            output_filename: Custom output filename (optional)
            
        Returns:
            pd.DataFrame: Features dataframe
        """
        try:
            # Load labels from file
            labels = load_labeled_patterns(labels_file)
            print(f"📖 Loaded {len(labels)} labeled patterns from file")
            
            # Extract features
            return self.extract_features_batch(labels, save_to_file, output_filename)
            
        except Exception as e:
            raise FeatureExtractionError(f"Failed to extract features from file: {e}")


def extract_features_from_labels(labels_file: Optional[str] = None,
                               output_file: Optional[str] = None,
                               window_size: int = DEFAULT_WINDOW_SIZE) -> pd.DataFrame:
    """
    Convenience function to extract features from labeled patterns.
    
    Args:
        labels_file: Path to labeled patterns JSON file
        output_file: Path to output CSV file
        window_size: Pattern window size in days
        
    Returns:
        pd.DataFrame: Extracted features
    """
    extractor = FeatureExtractor(window_size=window_size)
    return extractor.extract_features_from_file(labels_file, True, output_file) 