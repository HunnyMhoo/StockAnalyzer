"""
Feature Extraction Example

This script demonstrates how to use the FeatureExtractor class to extract
numerical features from labeled stock patterns for machine learning.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.feature_extractor import FeatureExtractor, extract_features_from_labels
    from src.pattern_labeler import PatternLabel, PatternLabeler
    from src.data_fetcher import fetch_hk_stocks
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


def main():
    """Main example function."""
    print("🔄 Feature Extraction Example")
    print("=" * 50)
    
    # Method 1: Extract features from existing labeled patterns file
    print("\n📖 Method 1: Extract from labeled patterns file")
    try:
        # Load and extract features from labeled patterns
        features_df = extract_features_from_labels(
            labels_file="labels/labeled_patterns.json",
            output_file="example_features.csv"
        )
        
        if not features_df.empty:
            print(f"✓ Extracted {len(features_df)} feature sets")
            print(f"✓ Features shape: {features_df.shape}")
            print(f"✓ Feature columns: {len(features_df.columns)} total")
            
            # Display sample features
            print("\n📊 Sample Features:")
            print(features_df.head())
            
            # Display feature summary
            print("\n📈 Feature Summary:")
            feature_cols = [col for col in features_df.columns 
                          if col not in ['ticker', 'start_date', 'end_date', 'label_type', 'notes']]
            print(f"Numerical features: {len(feature_cols)}")
            print("Feature categories:")
            
            trend_features = [col for col in feature_cols if 'trend' in col or 'sma' in col or 'angle' in col]
            correction_features = [col for col in feature_cols if 'drawdown' in col or 'recovery' in col or 'down_day' in col]
            support_features = [col for col in feature_cols if 'support' in col or 'break' in col]
            technical_features = [col for col in feature_cols if col in ['rsi_14', 'macd_diff', 'volatility', 'volume_avg_ratio']]
            
            print(f"  • Trend features: {len(trend_features)}")
            print(f"  • Correction features: {len(correction_features)}")
            print(f"  • Support break features: {len(support_features)}")
            print(f"  • Technical indicators: {len(technical_features)}")
            
        else:
            print("⚠️  No features extracted - check if data is available")
            
    except Exception as e:
        print(f"❌ Error with Method 1: {e}")
    
    # Method 2: Create FeatureExtractor instance for custom extraction
    print("\n\n🔧 Method 2: Custom FeatureExtractor usage")
    try:
        # Initialize feature extractor with custom parameters
        extractor = FeatureExtractor(
            window_size=25,  # Custom window size
            prior_context_days=35,  # Custom prior context
            support_lookback_days=12,  # Custom support lookback
            output_dir="features"
        )
        
        print(f"✓ Created FeatureExtractor with:")
        print(f"  • Window size: {extractor.window_size} days")
        print(f"  • Prior context: {extractor.prior_context_days} days")
        print(f"  • Support lookback: {extractor.support_lookback_days} days")
        
        # Create a sample pattern label for demonstration
        sample_label = PatternLabel(
            ticker="0700.HK",
            start_date="2023-02-10",
            end_date="2023-03-03",
            label_type="positive",
            notes="Example pattern for feature extraction"
        )
        
        # Extract features from single label
        print(f"\n🎯 Extracting features from sample pattern:")
        print(f"  • Ticker: {sample_label.ticker}")
        print(f"  • Period: {sample_label.start_date} to {sample_label.end_date}")
        print(f"  • Type: {sample_label.label_type}")
        
        features = extractor.extract_features_from_label(sample_label)
        
        if features:
            print("✓ Successfully extracted features")
            
            # Display features by category
            print("\n📋 Extracted Features by Category:")
            
            # Trend features
            trend_keys = ['prior_trend_return', 'above_sma_50_ratio', 'trend_angle']
            print("\n🔺 Trend Context Features:")
            for key in trend_keys:
                if key in features:
                    print(f"  • {key}: {features[key]:.4f}")
            
            # Correction features
            correction_keys = ['drawdown_pct', 'recovery_return_pct', 'down_day_ratio']
            print("\n📉 Correction Phase Features:")
            for key in correction_keys:
                if key in features:
                    print(f"  • {key}: {features[key]:.4f}")
            
            # Support break features
            support_keys = ['support_level', 'support_break_depth_pct', 'false_break_flag', 'recovery_days']
            print("\n🛡️  Support Break Features:")
            for key in support_keys:
                if key in features:
                    print(f"  • {key}: {features[key]:.4f}")
            
            # Technical indicators
            technical_keys = ['sma_5', 'sma_10', 'sma_20', 'rsi_14', 'macd_diff', 'volatility']
            print("\n📊 Technical Indicators:")
            for key in technical_keys:
                if key in features:
                    print(f"  • {key}: {features[key]:.4f}")
                    
        else:
            print("⚠️  Could not extract features - check if data is available")
            
    except Exception as e:
        print(f"❌ Error with Method 2: {e}")
    
    # Method 3: Batch processing with custom labels
    print("\n\n📦 Method 3: Batch processing demonstration")
    try:
        # Create multiple sample labels
        sample_labels = [
            PatternLabel("0700.HK", "2023-02-10", "2023-03-03", "positive", "Sample 1"),
            PatternLabel("0005.HK", "2022-10-15", "2022-11-01", "positive", "Sample 2"),
            PatternLabel("0001.HK", "2023-01-15", "2023-02-05", "positive", "Sample 3"),
        ]
        
        print(f"📋 Processing {len(sample_labels)} sample patterns...")
        
        # Extract features in batch
        batch_df = extractor.extract_features_batch(
            sample_labels, 
            save_to_file=True,
            output_filename="batch_example_features.csv"
        )
        
        if not batch_df.empty:
            print(f"✓ Batch processing completed")
            print(f"✓ Processed {len(batch_df)} patterns successfully")
            print(f"✓ Output saved to: features/batch_example_features.csv")
            
            # Show success rate by label type
            if 'label_type' in batch_df.columns:
                type_counts = batch_df['label_type'].value_counts()
                print(f"\n📈 Success by label type:")
                for label_type, count in type_counts.items():
                    print(f"  • {label_type}: {count} patterns")
                    
        else:
            print("⚠️  Batch processing yielded no results")
            
    except Exception as e:
        print(f"❌ Error with Method 3: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Feature extraction example completed!")
    print("\nNext steps:")
    print("1. Review the generated CSV files in the features/ directory")
    print("2. Use the extracted features for machine learning model training")
    print("3. Customize the FeatureExtractor parameters for your specific needs")
    print("4. Add more labeled patterns to improve feature diversity")


def demonstrate_feature_validation():
    """Demonstrate feature validation and quality checks."""
    print("\n🔍 Feature Validation Example")
    print("-" * 30)
    
    # Load example features if available
    feature_files = ["features/labeled_features.csv", "features/example_features.csv"]
    
    for file_path in feature_files:
        if os.path.exists(file_path):
            print(f"\n📄 Analyzing: {file_path}")
            
            try:
                df = pd.read_csv(file_path)
                
                # Basic statistics
                print(f"  • Shape: {df.shape}")
                print(f"  • Patterns: {len(df)}")
                
                # Feature quality checks
                numeric_cols = df.select_dtypes(include=[float, int]).columns
                print(f"  • Numeric features: {len(numeric_cols)}")
                
                # Check for missing values
                missing_counts = df[numeric_cols].isnull().sum()
                if missing_counts.sum() > 0:
                    print(f"  ⚠️  Missing values found in {missing_counts[missing_counts > 0].shape[0]} features")
                else:
                    print("  ✓ No missing values")
                
                # Check for infinite values
                inf_counts = pd.DataFrame(df[numeric_cols]).applymap(lambda x: not pd.isfinite(x) if pd.notnull(x) else False).sum()
                if inf_counts.sum() > 0:
                    print(f"  ⚠️  Infinite values found in {inf_counts[inf_counts > 0].shape[0]} features")
                else:
                    print("  ✓ No infinite values")
                
                # Feature statistics
                print(f"  • Feature ranges:")
                for col in numeric_cols[:5]:  # Show first 5 features
                    min_val, max_val = df[col].min(), df[col].max()
                    print(f"    - {col}: [{min_val:.2f}, {max_val:.2f}]")
                
                if len(numeric_cols) > 5:
                    print(f"    ... and {len(numeric_cols) - 5} more features")
                    
            except Exception as e:
                print(f"  ❌ Error analyzing {file_path}: {e}")
                
            break  # Only analyze first available file
    else:
        print("  ⚠️  No feature files found to analyze")
        print("  💡 Run feature extraction first to generate example files")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Run feature validation demo
    demonstrate_feature_validation() 