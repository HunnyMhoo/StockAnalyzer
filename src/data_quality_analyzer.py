"""
Data Quality Analysis Module

This module provides utilities for analyzing stock data quality, availability,
and readiness for pattern scanning. Extracted from notebook utilities for
better reusability across the codebase.

Key Components:
- DataQualityAnalyzer: Main analysis class
- DataQualitySummary: Results container
- Stock availability checking
- Data freshness validation
- Quality scoring metrics
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.data_fetcher import validate_cached_data_file, get_all_cached_tickers


@dataclass
class DataQualitySummary:
    """Container for data quality analysis results"""
    total_stocks: int
    quality_distribution: Dict[str, int]
    quality_percentages: Dict[str, float]
    sample_stocks: List[str]
    cache_status: Dict[str, Any]
    scanning_readiness: bool
    recommendations: List[str]


class DataQualityAnalyzer:
    """
    Analyzer for stock data quality and availability assessment.
    
    Provides functionality to:
    1. Assess cached data quality across stock universe
    2. Generate quality distribution metrics
    3. Provide scanning readiness recommendations
    4. Support data collection planning
    """
    
    def __init__(self, sample_size: int = 20):
        """
        Initialize the data quality analyzer.
        
        Args:
            sample_size: Number of stocks to sample for quality assessment
        """
        self.sample_size = sample_size
        
    def analyze_data_quality(self, show_details: bool = True) -> DataQualitySummary:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            show_details: Whether to print detailed analysis output
            
        Returns:
            DataQualitySummary: Complete quality analysis results
        """
        if show_details:
            print("🔍 Analyzing available stock data...")
            
        # Get available stocks
        available_stocks = get_all_cached_tickers()
        
        if not available_stocks:
            if show_details:
                print("❌ No cached stock data found!")
                print("📝 Please run data collection notebooks first:")
                print("   • 02_basic_data_collection.py (for beginners)")
                print("   • 02_advanced_data_collection.py (for large-scale)")
            
            return DataQualitySummary(
                total_stocks=0,
                quality_distribution={'high': 0, 'medium': 0, 'low': 0},
                quality_percentages={'high': 0.0, 'medium': 0.0, 'low': 0.0},
                sample_stocks=[],
                cache_status={'status': 'empty', 'last_update': None},
                scanning_readiness=False,
                recommendations=[
                    "Run data collection notebooks to gather stock data",
                    "Start with 02_basic_data_collection.py for initial setup"
                ]
            )
        
        if show_details:
            print(f"✅ Found {len(available_stocks)} stocks with cached data")
        
        # Perform quality assessment on sample
        quality_results = self._assess_sample_quality(available_stocks, show_details)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_results, len(available_stocks))
        
        # Create summary
        summary = DataQualitySummary(
            total_stocks=len(available_stocks),
            quality_distribution=quality_results['distribution'],
            quality_percentages=quality_results['percentages'],
            sample_stocks=available_stocks[:10],  # Show first 10 as sample
            cache_status=self._get_cache_status(),
            scanning_readiness=quality_results['percentages']['high'] >= 0.5,  # 50% high quality threshold
            recommendations=recommendations
        )
        
        if show_details:
            self._display_quality_summary(summary)
            
        return summary
    
    def _assess_sample_quality(self, available_stocks: List[str], show_details: bool) -> Dict[str, Any]:
        """
        Assess quality of a sample of available stocks.
        
        Args:
            available_stocks: List of available stock tickers
            show_details: Whether to show detailed output
            
        Returns:
            Dict containing quality assessment results
        """
        sample_size = min(self.sample_size, len(available_stocks))
        sample_stocks = available_stocks[:sample_size]
        
        high_quality = medium_quality = low_quality = 0
        
        if show_details:
            print(f"📊 Quality assessment (sample of {sample_size} stocks):")
        
        for ticker in sample_stocks:
            try:
                validation = validate_cached_data_file(ticker)
                quality_score = validation.get('data_quality_score', 0)
                
                if quality_score >= 0.8:
                    high_quality += 1
                elif quality_score >= 0.6:
                    medium_quality += 1
                else:
                    low_quality += 1
            except Exception:
                low_quality += 1
        
        # Calculate percentages
        total_sample = high_quality + medium_quality + low_quality
        percentages = {
            'high': high_quality / total_sample * 100 if total_sample > 0 else 0,
            'medium': medium_quality / total_sample * 100 if total_sample > 0 else 0,
            'low': low_quality / total_sample * 100 if total_sample > 0 else 0
        }
        
        if show_details:
            print(f"   🟢 High Quality: {high_quality}/{sample_size} stocks ({percentages['high']:.0f}%)")
            print(f"   🟡 Medium Quality: {medium_quality}/{sample_size} stocks ({percentages['medium']:.0f}%)")
            print(f"   🔴 Low Quality: {low_quality}/{sample_size} stocks ({percentages['low']:.0f}%)")
        
        return {
            'distribution': {
                'high': high_quality,
                'medium': medium_quality,
                'low': low_quality
            },
            'percentages': {
                'high': percentages['high'] / 100,  # Convert to decimal
                'medium': percentages['medium'] / 100,
                'low': percentages['low'] / 100
            }
        }
    
    def _get_cache_status(self) -> Dict[str, Any]:
        """Get cache status information"""
        try:
            # Check if data directory exists and get modification time
            data_dir = "data"
            if os.path.exists(data_dir):
                csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                if csv_files:
                    # Get most recent file modification time
                    latest_mod_time = max(
                        os.path.getmtime(os.path.join(data_dir, f)) 
                        for f in csv_files
                    )
                    last_update = datetime.fromtimestamp(latest_mod_time)
                    
                    # Check if data is fresh (within 7 days)
                    days_old = (datetime.now() - last_update).days
                    status = "fresh" if days_old <= 7 else "stale"
                    
                    return {
                        'status': status,
                        'last_update': last_update,
                        'days_old': days_old,
                        'file_count': len(csv_files)
                    }
            
            return {'status': 'empty', 'last_update': None}
            
        except Exception:
            return {'status': 'error', 'last_update': None}
    
    def _generate_recommendations(self, quality_results: Dict[str, Any], total_stocks: int) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        high_pct = quality_results['percentages']['high']
        medium_pct = quality_results['percentages']['medium']
        low_pct = quality_results['percentages']['low']
        
        # Quality-based recommendations
        if high_pct >= 0.7:
            recommendations.append("✅ Excellent data quality - ready for pattern scanning")
        elif high_pct >= 0.5:
            recommendations.append("✅ Good data quality - suitable for most pattern analysis")
        elif medium_pct + high_pct >= 0.6:
            recommendations.append("⚠️ Acceptable data quality - consider filtering low-quality stocks")
        else:
            recommendations.append("❌ Poor data quality - recommend data collection refresh")
        
        # Stock count recommendations
        if total_stocks >= 100:
            recommendations.append("📊 Large stock universe - consider sector-based analysis")
        elif total_stocks >= 50:
            recommendations.append("📊 Good stock coverage - ready for comprehensive scanning")
        elif total_stocks >= 20:
            recommendations.append("📊 Moderate coverage - suitable for focused analysis")
        else:
            recommendations.append("📊 Limited coverage - consider expanding stock universe")
        
        # Cache freshness recommendations
        cache_status = self._get_cache_status()
        if cache_status.get('days_old', 0) > 7:
            recommendations.append("🔄 Data is stale - consider refreshing cache")
        
        return recommendations
    
    def _display_quality_summary(self, summary: DataQualitySummary) -> None:
        """Display formatted quality summary"""
        print(f"\n📈 Sample available stocks: {', '.join(summary.sample_stocks)}")
        if len(summary.sample_stocks) < summary.total_stocks:
            remaining = summary.total_stocks - len(summary.sample_stocks)
            print(f"   ... and {remaining} more")
        
        print(f"\n🎯 Ready to scan {summary.total_stocks} stocks for patterns!")
        
        # Show recommendations
        if summary.recommendations:
            print(f"\n💡 **Recommendations:**")
            for rec in summary.recommendations:
                print(f"   {rec}")
    
    def get_scanning_readiness_report(self) -> Dict[str, Any]:
        """
        Get a focused report on scanning readiness.
        
        Returns:
            Dict with scanning readiness metrics
        """
        summary = self.analyze_data_quality(show_details=False)
        
        return {
            'ready_for_scanning': summary.scanning_readiness,
            'total_stocks': summary.total_stocks,
            'quality_score': summary.quality_percentages['high'] + summary.quality_percentages['medium'] * 0.5,
            'recommendations': summary.recommendations,
            'cache_status': summary.cache_status
        }


def show_enhanced_data_summary() -> List[str]:
    """
    Enhanced data summary with quality metrics and scanning readiness.
    
    This function maintains the same interface as the original notebook function
    but now uses the refactored DataQualityAnalyzer.
    
    Returns:
        List of available stock tickers
    """
    analyzer = DataQualityAnalyzer(sample_size=20)
    summary = analyzer.analyze_data_quality(show_details=True)
    
    # Return the available stocks for backward compatibility
    return get_all_cached_tickers()


def quick_quality_check() -> bool:
    """
    Quick check if data is ready for pattern scanning.
    
    Returns:
        bool: True if data is ready for scanning
    """
    analyzer = DataQualityAnalyzer(sample_size=10)
    report = analyzer.get_scanning_readiness_report()
    return report['ready_for_scanning'] 