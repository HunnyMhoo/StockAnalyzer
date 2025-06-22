"""
Pattern Scanning Example

This script demonstrates how to use the PatternScanner to detect trading patterns
across multiple Hong Kong stocks using trained machine learning models.
"""

import os
import sys
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pattern_scanner import (
    PatternScanner, ScanningConfig, scan_hk_stocks_for_patterns
)
from hk_stock_universe import get_top_hk_stocks, MAJOR_HK_STOCKS


def example_1_basic_scanning():
    """
    Example 1: Basic pattern scanning with default configuration.
    """
    print("=" * 60)
    print("Example 1: Basic Pattern Scanning")
    print("=" * 60)
    
    # Find the most recent model file
    models_dir = "../models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("❌ No model files found in ./models/ directory")
        print("   Please run model training first (Story 1.4)")
        return
    
    # Use the most recent model
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    
    print(f"📖 Using model: {latest_model}")
    
    # Use a small subset of HK stocks for demonstration
    demo_tickers = ['0700.HK', '0005.HK', '0388.HK', '1299.HK', '0941.HK']
    
    try:
        # Run scanning with convenience function
        results = scan_hk_stocks_for_patterns(
            model_path=model_path,
            ticker_list=demo_tickers,
            window_size=30,
            min_confidence=0.65,
            max_windows_per_ticker=3
        )
        
        print(f"\n📊 Scanning Results Summary:")
        print(f"  • Total matches found: {len(results.matches_df)}")
        print(f"  • Scanning time: {results.scanning_time:.2f} seconds")
        print(f"  • Average confidence: {results.scanning_summary.get('average_confidence', 0):.3f}")
        
        if len(results.matches_df) > 0:
            print(f"\n🎯 Top Matches:")
            for _, row in results.matches_df.head(3).iterrows():
                print(f"  {row['rank']:2d}. {row['ticker']:8s} | "
                      f"{row['window_start_date']} to {row['window_end_date']} | "
                      f"Confidence: {row['confidence_score']:.3f}")
        else:
            print("\n⚠️  No pattern matches found above confidence threshold")
            print("   Try lowering the min_confidence parameter")
        
    except Exception as e:
        print(f"❌ Error during scanning: {e}")
        print("   Please check that you have data files in ./data/ directory")


def example_2_custom_configuration():
    """
    Example 2: Pattern scanning with custom configuration.
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration Scanning")
    print("=" * 60)
    
    # Find model file
    models_dir = "../models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("❌ No model files found")
        return
    
    model_path = os.path.join(models_dir, sorted(model_files)[-1])
    
    try:
        # Initialize PatternScanner with custom configuration
        scanner = PatternScanner(
            model_path=model_path,
            feature_extractor_config={
                'window_size': 25,  # Custom window size
                'prior_context_days': 35,  # Custom prior context
                'support_lookback_days': 15  # Custom support lookback
            }
        )
        
        # Create custom scanning configuration
        config = ScanningConfig(
            window_size=25,  # Match feature extractor
            min_confidence=0.75,  # Higher confidence threshold
            max_windows_per_ticker=2,  # Fewer windows per ticker
            top_matches_display=10,
            save_results=True,
            output_filename=f"custom_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            include_feature_values=True  # Include feature values in output
        )
        
        # Use technology stocks for scanning
        tech_tickers = MAJOR_HK_STOCKS['tech_stocks'][:8]  # First 8 tech stocks
        
        print(f"🔍 Scanning {len(tech_tickers)} technology stocks:")
        print(f"  Tickers: {', '.join(tech_tickers)}")
        print(f"  Window size: {config.window_size} days")
        print(f"  Min confidence: {config.min_confidence}")
        
        # Run scanning
        results = scanner.scan_tickers(tech_tickers, config)
        
        # Display detailed results
        print(f"\n📈 Technology Stocks Scan Results:")
        print(f"  • Model used: {results.model_info['model_type']}")
        print(f"  • Tickers scanned: {results.scanning_summary['total_tickers_scanned']}")
        print(f"  • High-confidence matches: {len(results.matches_df)}")
        
        if results.output_path:
            print(f"  • Results saved to: {results.output_path}")
        
        # Show best matches with feature values if available
        if len(results.matches_df) > 0 and config.include_feature_values:
            print(f"\n🎯 Best Match Details:")
            best_match = results.matches_df.iloc[0]
            print(f"  Ticker: {best_match['ticker']}")
            print(f"  Period: {best_match['window_start_date']} to {best_match['window_end_date']}")
            print(f"  Confidence: {best_match['confidence_score']:.3f}")
            
            # Show key features if included
            feature_cols = [col for col in results.matches_df.columns if col.startswith('feature_')]
            if feature_cols:
                print(f"  Key Features:")
                for col in feature_cols[:5]:  # Show first 5 features
                    feature_name = col.replace('feature_', '')
                    print(f"    {feature_name}: {best_match[col]:.4f}")
        
    except Exception as e:
        print(f"❌ Error during custom scanning: {e}")


def example_3_sector_analysis():
    """
    Example 3: Sector-based pattern analysis.
    """
    print("\n" + "=" * 60)
    print("Example 3: Sector-Based Pattern Analysis")
    print("=" * 60)
    
    # Find model file
    models_dir = "../models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("❌ No model files found")
        return
    
    model_path = os.path.join(models_dir, sorted(model_files)[-1])
    
    try:
        scanner = PatternScanner(model_path)
        
        # Configuration for sector analysis
        config = ScanningConfig(
            window_size=30,
            min_confidence=0.60,  # Lower threshold to catch more patterns
            max_windows_per_ticker=4,
            save_results=False  # Don't save individual sector results
        )
        
        sectors = ['tech_stocks', 'finance', 'property']
        sector_results = {}
        
        print("🏢 Analyzing patterns by sector:")
        
        for sector in sectors:
            if sector in MAJOR_HK_STOCKS:
                tickers = MAJOR_HK_STOCKS[sector][:6]  # Limit to 6 stocks per sector
                
                print(f"\n  📊 {sector.replace('_', ' ').title()} Sector:")
                print(f"     Scanning: {', '.join(tickers)}")
                
                results = scanner.scan_tickers(tickers, config)
                sector_results[sector] = results
                
                matches = len(results.matches_df)
                avg_confidence = results.scanning_summary.get('average_confidence', 0)
                
                print(f"     Matches found: {matches}")
                print(f"     Average confidence: {avg_confidence:.3f}")
                
                if matches > 0:
                    best_ticker = results.matches_df.iloc[0]['ticker']
                    best_confidence = results.matches_df.iloc[0]['confidence_score']
                    print(f"     Best match: {best_ticker} ({best_confidence:.3f})")
        
        # Compare sectors
        print(f"\n🏆 Sector Pattern Summary:")
        for sector, results in sector_results.items():
            matches = len(results.matches_df)
            avg_conf = results.scanning_summary.get('average_confidence', 0)
            max_conf = results.scanning_summary.get('max_confidence', 0)
            
            print(f"  {sector.replace('_', ' ').title():12s}: "
                  f"{matches:2d} matches, "
                  f"avg: {avg_conf:.3f}, "
                  f"max: {max_conf:.3f}")
        
        # Find the sector with the most patterns
        best_sector = max(sector_results.keys(), 
                         key=lambda s: len(sector_results[s].matches_df))
        
        if len(sector_results[best_sector].matches_df) > 0:
            print(f"\n🎯 Most Active Sector: {best_sector.replace('_', ' ').title()}")
            top_matches = sector_results[best_sector].matches_df.head(3)
            for _, match in top_matches.iterrows():
                print(f"  • {match['ticker']} | "
                      f"{match['window_start_date']} to {match['window_end_date']} | "
                      f"Confidence: {match['confidence_score']:.3f}")
        
    except Exception as e:
        print(f"❌ Error during sector analysis: {e}")


def example_4_performance_test():
    """
    Example 4: Performance testing with larger stock universe.
    """
    print("\n" + "=" * 60)
    print("Example 4: Performance Testing")
    print("=" * 60)
    
    # Find model file
    models_dir = "../models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("❌ No model files found")
        return
    
    model_path = os.path.join(models_dir, sorted(model_files)[-1])
    
    try:
        # Get larger set of tickers for performance testing
        all_major_tickers = []
        for category, tickers in MAJOR_HK_STOCKS.items():
            all_major_tickers.extend(tickers)
        
        # Remove duplicates
        unique_tickers = list(set(all_major_tickers))
        
        # Test with increasing numbers of tickers
        test_sizes = [5, 10, 20, min(30, len(unique_tickers))]
        
        print(f"🚀 Performance test with increasing ticker counts:")
        print(f"   Available tickers: {len(unique_tickers)}")
        
        scanner = PatternScanner(model_path)
        
        config = ScanningConfig(
            window_size=30,
            min_confidence=0.70,
            max_windows_per_ticker=3,
            save_results=False
        )
        
        for size in test_sizes:
            test_tickers = unique_tickers[:size]
            
            print(f"\n  📊 Testing with {size} tickers...")
            start_time = datetime.now()
            
            results = scanner.scan_tickers(test_tickers, config)
            
            end_time = datetime.now()
            actual_time = (end_time - start_time).total_seconds()
            
            # Performance metrics
            tickers_per_second = size / actual_time if actual_time > 0 else 0
            matches_found = len(results.matches_df)
            success_rate = results.scanning_summary['total_tickers_scanned'] / size * 100
            
            print(f"     Time: {actual_time:.2f}s | "
                  f"Rate: {tickers_per_second:.1f} tickers/s | "
                  f"Success: {success_rate:.1f}% | "
                  f"Matches: {matches_found}")
            
            # Check if we meet performance requirements
            if size >= 20 and actual_time <= 60:
                print(f"     ✅ Performance requirement met ({size} tickers in {actual_time:.1f}s)")
            elif size >= 20:
                print(f"     ⚠️  Performance below target (target: <60s for {size}+ tickers)")
        
        print(f"\n📈 Performance Summary:")
        print(f"  • Model loading and initialization is done once")
        print(f"  • Scanning time scales roughly linearly with ticker count")
        print(f"  • Feature extraction is the main performance bottleneck")
        print(f"  • Consider using fewer windows per ticker for larger scans")
        
    except Exception as e:
        print(f"❌ Error during performance test: {e}")


def main():
    """
    Main function to run all examples.
    """
    print("🔍 Pattern Scanner Examples")
    print("This script demonstrates various ways to use the PatternScanner")
    print("for detecting trading patterns in Hong Kong stocks.")
    print()
    
    # Check prerequisites
    if not os.path.exists("../models"):
        print("❌ Models directory not found. Please ensure you're running from the examples/ directory")
        print("   and that you have trained models in the models/ directory")
        return
    
    if not os.path.exists("../data"):
        print("❌ Data directory not found. Please ensure you have stock data files")
        print("   Run the bulk data fetcher first (Story 1.2)")
        return
    
    try:
        # Run all examples
        example_1_basic_scanning()
        example_2_custom_configuration()
        example_3_sector_analysis()
        example_4_performance_test()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\nNext steps:")
        print("  • Check the signals/ directory for saved results")
        print("  • Experiment with different confidence thresholds")
        print("  • Try different window sizes and sector combinations")
        print("  • Use the pattern scanner in your own trading strategy")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("   Please check your model files and data availability")


if __name__ == "__main__":
    main() 