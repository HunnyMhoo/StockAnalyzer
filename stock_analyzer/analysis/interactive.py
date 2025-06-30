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
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
from dataclasses import dataclass

# Import core modules
from ..features.extractor import FeatureExtractor
from ..data.fetcher import fetch_hk_stocks, get_all_cached_tickers


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
    # Enhanced negative sampling options
    negative_sampling_strategy: str = "temporal_recent"  # "temporal_recent", "multiple_windows", "random"
    negative_samples_per_ticker: int = 4
    # Enhanced stock selection options
    stock_selection_strategy: str = "random_diverse"  # "random_diverse", "alphabetical", "market_cap_balanced"


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
        ENHANCED: Generate more robust training data to avoid 25% confidence issue.
        
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
        
        # ENHANCEMENT 1: Generate multiple positive variations through data augmentation
        all_features = []
        all_labels = []
        
        # Add original positive example
        all_features.append(positive_features)
        all_labels.append(1)
        
        # ENHANCEMENT 2: Create slight variations of positive pattern (data augmentation)
        for i in range(2):  # Generate 2 additional positive variations
            augmented_features = self._augment_features(positive_features, noise_level=0.05)
            all_features.append(augmented_features)
            all_labels.append(1)
        
        # Extract features for negative examples with enhanced sampling
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
        
        # ENHANCEMENT 3: Generate more negative samples per ticker
        min_negative_samples = max(8, len(negative_tickers) * 4)  # At least 8 negative samples
        
        for neg_ticker, neg_df in negative_data.items():
            if len(neg_df) > pattern_length + config.context_days:
                # Enhanced negative sampling strategy
                negative_windows = self._generate_negative_windows(
                    neg_df, pattern_length, positive_example, config
                )
                
                # Ensure we get enough negative samples from each ticker
                samples_per_ticker = max(4, min_negative_samples // len(negative_tickers))
                
                for idx, (neg_window, neg_context) in enumerate(negative_windows):
                    if idx >= samples_per_ticker:
                        break
                        
                    neg_features = self.feature_extractor.extract_features_from_window_data(
                        neg_window, neg_context, neg_ticker, 
                        str(neg_window.index.min().date()), 
                        str(neg_window.index.max().date()), 
                        neg_df
                    )
                    
                    if neg_features:
                        all_features.append(neg_features)
                        all_labels.append(0)
        
        # ENHANCEMENT 4: Balance classes for better training
        positive_count = all_labels.count(1)
        negative_count = all_labels.count(0)
        
        if not config.quiet_mode:
            print(f"🔍 Enhanced training data: {positive_count} positive, {negative_count} negative samples")
        
        if negative_count == 0:
            raise ValueError("Failed to generate negative training samples from the provided tickers")
        
        # ENHANCEMENT 5: Add synthetic negative examples if still too few
        if negative_count < positive_count * 2:  # Want at least 2:1 negative:positive ratio
            additional_negatives_needed = (positive_count * 2) - negative_count
            for i in range(additional_negatives_needed):
                # Create synthetic negative by significantly modifying a positive example
                synthetic_negative = self._create_synthetic_negative(positive_features)
                all_features.append(synthetic_negative)
                all_labels.append(0)
        
        # Get feature names (exclude metadata columns)
        full_features_df = pd.DataFrame(all_features)
        metadata_cols = ['ticker', 'start_date', 'end_date', 'label_type', 'notes']
        feature_names = [col for col in full_features_df.columns if col not in metadata_cols]
        
        final_positive = all_labels.count(1)
        final_negative = all_labels.count(0)
        
        if not config.quiet_mode:
            print(f"🎯 Final training set: {final_positive} positive, {final_negative} negative ({len(all_features)} total)")
        
        return all_features, all_labels, feature_names
    
    def _augment_features(self, features: Dict[str, Union[str, float]], noise_level: float = 0.05) -> Dict[str, Union[str, float]]:
        """
        Create augmented version of features by adding small random noise.
        
        Args:
            features: Original features dictionary
            noise_level: Standard deviation of noise to add (as fraction of feature value)
            
        Returns:
            Augmented features dictionary
        """
        augmented = features.copy()
        
        for key, value in features.items():
            # Only augment numeric features, skip metadata
            if key not in ['ticker', 'start_date', 'end_date', 'label_type', 'notes']:
                try:
                    numeric_value = float(value)
                    # Add small random noise
                    noise = np.random.normal(0, noise_level * abs(numeric_value))
                    augmented[key] = numeric_value + noise
                except (ValueError, TypeError):
                    # Keep non-numeric values unchanged
                    augmented[key] = value
        
        return augmented
    
    def _create_synthetic_negative(self, positive_features: Dict[str, Union[str, float]]) -> Dict[str, Union[str, float]]:
        """
        Create a synthetic negative example by significantly modifying positive features.
        
        Args:
            positive_features: Original positive pattern features
            
        Returns:
            Synthetic negative features dictionary
        """
        synthetic = positive_features.copy()
        
        # Modify key features that distinguish positive patterns
        key_features = [
            'prior_trend_return', 'drawdown_pct', 'recovery_return_pct', 
            'break_depth_pct', 'rsi_avg', 'macd_signal_avg'
        ]
        
        for key, value in positive_features.items():
            if key not in ['ticker', 'start_date', 'end_date', 'label_type', 'notes']:
                try:
                    numeric_value = float(value)
                    
                    if key in key_features:
                        # Significantly modify key features (flip signs, change magnitude)
                        if key in ['prior_trend_return', 'recovery_return_pct']:
                            # Flip positive trends to negative
                            synthetic[key] = -abs(numeric_value) - 0.02
                        elif key == 'drawdown_pct':
                            # Make drawdown much smaller (less significant dip)
                            synthetic[key] = numeric_value * 0.3
                        elif key == 'break_depth_pct':
                            # Reduce break depth (less breakout strength)
                            synthetic[key] = numeric_value * 0.5
                        elif key in ['rsi_avg', 'macd_signal_avg']:
                            # Invert momentum indicators
                            synthetic[key] = 100 - numeric_value if key == 'rsi_avg' else -numeric_value
                        else:
                            # General inversion for other key features
                            synthetic[key] = -numeric_value
                    else:
                        # Add moderate noise to other features
                        noise = np.random.normal(0, 0.1 * abs(numeric_value))
                        synthetic[key] = numeric_value + noise
                        
                except (ValueError, TypeError):
                    # Keep non-numeric values unchanged
                    synthetic[key] = value
        
        return synthetic
    
    def _generate_negative_windows(self, neg_df: pd.DataFrame, pattern_length: int, 
                                 positive_example: Dict, config: PatternAnalysisConfig) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate negative example windows using improved sampling strategies.
        
        Args:
            neg_df: Negative ticker DataFrame
            pattern_length: Length of the positive pattern
            positive_example: Positive pattern information
            config: Analysis configuration
            
        Returns:
            List of (negative_window, negative_context) tuples
        """
        windows = []
        
        if config.negative_sampling_strategy == "temporal_recent":
            # Sample from recent periods (similar time frame as positive example)
            positive_end = positive_example['end_date']
            
            # Try to get windows within 3 months of positive example
            recent_start = positive_end - timedelta(days=90)
            recent_end = positive_end + timedelta(days=30)
            
            # Filter to recent period if data exists
            recent_data = neg_df.loc[recent_start:recent_end] if recent_start in neg_df.index or recent_end in neg_df.index else neg_df
            
            if len(recent_data) > pattern_length + config.context_days:
                # Generate multiple windows from recent period
                for i in range(config.negative_samples_per_ticker):
                    if len(recent_data) > pattern_length + config.context_days:
                        max_start = len(recent_data) - pattern_length - config.context_days
                        if max_start > 0:
                            rand_start = np.random.randint(0, max_start)
                            neg_window = recent_data.iloc[rand_start + config.context_days : rand_start + config.context_days + pattern_length]
                            neg_context = recent_data.iloc[rand_start : rand_start + config.context_days]
                            windows.append((neg_window, neg_context))
        
        elif config.negative_sampling_strategy == "multiple_windows":
            # Generate multiple diverse windows across different time periods
            total_data_length = len(neg_df)
            segment_length = total_data_length // config.negative_samples_per_ticker
            
            for i in range(config.negative_samples_per_ticker):
                segment_start = i * segment_length
                segment_end = min((i + 1) * segment_length, total_data_length - pattern_length - config.context_days)
                
                if segment_end > segment_start + pattern_length + config.context_days:
                    rand_start = np.random.randint(segment_start, segment_end)
                    neg_window = neg_df.iloc[rand_start + config.context_days : rand_start + config.context_days + pattern_length]
                    neg_context = neg_df.iloc[rand_start : rand_start + config.context_days]
                    windows.append((neg_window, neg_context))
        
        else:  # "random" - fallback to original behavior
            for i in range(config.negative_samples_per_ticker):
                rand_start = np.random.randint(0, len(neg_df) - pattern_length - config.context_days)
                neg_window = neg_df.iloc[rand_start + config.context_days : rand_start + config.context_days + pattern_length]
                neg_context = neg_df.iloc[rand_start : rand_start + config.context_days]
                windows.append((neg_window, neg_context))
        
        return windows
    
    def _select_diverse_stocks(self, stock_list: List[str], max_stocks: int, 
                             config: PatternAnalysisConfig) -> List[str]:
        """
        Select a diverse subset of stocks using improved selection strategies.
        
        Args:
            stock_list: Full list of available stocks
            max_stocks: Maximum number of stocks to select
            config: Analysis configuration
            
        Returns:
            Selected subset of stocks
        """
        if len(stock_list) <= max_stocks:
            return stock_list
        
        if config.stock_selection_strategy == "random_diverse":
            # Random sampling for fair representation across all stocks
            import random
            selected = random.sample(stock_list, max_stocks)
            return sorted(selected)  # Sort for consistent output
            
        elif config.stock_selection_strategy == "market_cap_balanced":
            # Try to balance by stock number patterns (rough proxy for listing order/market cap)
            # Group by first digit pattern: 0xxx, 1xxx, 2xxx, etc.
            from collections import defaultdict
            groups = defaultdict(list)
            
            for ticker in stock_list:
                # Extract first digit of stock number
                stock_number = ticker.split('.')[0]
                if stock_number.isdigit():
                    first_digit = stock_number[0]
                    groups[first_digit].append(ticker)
            
            # Distribute stocks proportionally across groups
            selected = []
            group_keys = sorted(groups.keys())
            stocks_per_group = max_stocks // len(group_keys)
            remainder = max_stocks % len(group_keys)
            
            for i, group_key in enumerate(group_keys):
                group_stocks = groups[group_key]
                group_limit = stocks_per_group + (1 if i < remainder else 0)
                
                if len(group_stocks) <= group_limit:
                    selected.extend(group_stocks)
                else:
                    import random
                    selected.extend(random.sample(group_stocks, group_limit))
                    
                if len(selected) >= max_stocks:
                    break
            
            return sorted(selected[:max_stocks])
            
        else:  # "alphabetical" - fallback to original behavior
            return stock_list[:max_stocks]
    
    def _train_temporary_model(self, all_features: List[Dict], all_labels: List[int], 
                             feature_names: List[str], config: PatternAnalysisConfig) -> Dict[str, Any]:
        """
        Train a temporary model for pattern detection.
        ENHANCED: Improved training with better regularization and feature scaling.
        
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
        
        # ENHANCEMENT 1: Feature scaling for better model performance
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        training_df_scaled = pd.DataFrame(
            scaler.fit_transform(training_df),
            columns=training_df.columns
        )
        
        # Only show training info if not in quiet mode
        if not config.quiet_mode:
            print(f"🤖 Enhanced training on {len(training_df_scaled)} samples "
                  f"({all_labels.count(1)} positive, {all_labels.count(0)} negative)...")
            print(f"🔧 Applied feature scaling and regularization")
        
        try:
            import xgboost as xgb
            import joblib
            
            # ENHANCEMENT 2: Better XGBoost parameters for small datasets
            model = xgb.XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss',
                # Regularization for small datasets
                max_depth=3,           # Prevent overfitting
                learning_rate=0.1,     # Conservative learning
                n_estimators=50,       # Fewer trees for small data
                subsample=0.8,         # Row sampling
                colsample_bytree=0.8,  # Feature sampling
                reg_alpha=1.0,         # L1 regularization
                reg_lambda=1.0,        # L2 regularization
                random_state=42        # Reproducible results
            )
            
            model.fit(training_df_scaled, pd.Series(all_labels))
            
            # ENHANCEMENT 3: Validate model performance on training data
            training_predictions = model.predict_proba(training_df_scaled)
            training_confidence_avg = training_predictions[:, 1].mean()
            training_confidence_std = training_predictions[:, 1].std()
            
            if not config.quiet_mode:
                print(f"📊 Training confidence - Mean: {training_confidence_avg:.1%}, "
                      f"Std: {training_confidence_std:.3f}")
            
            # Warning if model still shows poor performance
            if training_confidence_std < 0.1:
                if not config.quiet_mode:
                    print("⚠️  Warning: Low confidence variance detected. Results may be unreliable.")
            
            # ENHANCEMENT 4: Fallback to Random Forest for very small datasets
            final_model = model
            final_model_type = 'enhanced_xgboost'
            
            # If XGBoost shows poor performance (low variance), try Random Forest
            if training_confidence_std < 0.05 or len(training_df_scaled) < 10:
                if not config.quiet_mode:
                    print("🔄 Switching to Random Forest for small dataset...")
                
                from sklearn.ensemble import RandomForestClassifier
                
                rf_model = RandomForestClassifier(
                    n_estimators=20,      # Fewer trees for small data
                    max_depth=4,          # Limited depth
                    min_samples_split=2,  # Minimum samples to split
                    min_samples_leaf=1,   # Minimum samples in leaf
                    bootstrap=True,       # Bootstrap sampling
                    random_state=42,      # Reproducible
                    class_weight='balanced'  # Handle class imbalance
                )
                
                rf_model.fit(training_df_scaled, pd.Series(all_labels))
                rf_predictions = rf_model.predict_proba(training_df_scaled)
                rf_confidence_avg = rf_predictions[:, 1].mean()
                rf_confidence_std = rf_predictions[:, 1].std()
                
                if not config.quiet_mode:
                    print(f"🌲 Random Forest confidence - Mean: {rf_confidence_avg:.1%}, "
                          f"Std: {rf_confidence_std:.3f}")
                
                # Use Random Forest if it shows better variance
                if rf_confidence_std > training_confidence_std:
                    final_model = rf_model
                    final_model_type = 'random_forest'
                    training_confidence_avg = rf_confidence_avg
                    training_confidence_std = rf_confidence_std
                    if not config.quiet_mode:
                        print("✅ Random Forest selected for better performance")
                else:
                    if not config.quiet_mode:
                        print("📊 Keeping XGBoost as primary model")
            
            # Create model package with scaler
            model_package = {
                'model': final_model,
                'scaler': scaler,  # Include scaler for prediction time
                'feature_names': feature_names,
                'config': SimplePatternConfig(),
                'metadata': {
                    'training_date': datetime.now().isoformat(),
                    'n_samples': len(training_df_scaled),
                    'n_features': len(feature_names),
                    'class_distribution': pd.Series(all_labels).value_counts().to_dict(),
                    'training_confidence_avg': training_confidence_avg,
                    'training_confidence_std': training_confidence_std,
                    'model_type': final_model_type
                }
            }
            
            # Save model package
            joblib.dump(model_package, self.temp_model_path)
            return model_package
            
        except Exception as e:
            raise Exception(f"Enhanced model training failed: {e}")
    
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
        
        # Apply scan limit with improved stock selection strategy
        if config.max_stocks_to_scan and config.max_stocks_to_scan < len(scan_list):
            scan_list = self._select_diverse_stocks(scan_list, config.max_stocks_to_scan, config)
        
        # Only show summary info in quiet mode, no detailed discovery
        if not config.quiet_mode:
            print(f"📊 Found {len(all_available_stocks)} stocks with cached data")
            print(f"🔎 Scanning {len(scan_list)} stocks for similar patterns...")
        
        # Initialize scanner and run scan (import locally to avoid circular dependency)
        from ..patterns.scanner import PatternScanner, ScanningConfig
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