"""
Improved Bulk Collection Utilities

This module consolidates and improves the bulk data collection functionality
from the original notebook, eliminating redundancy and improving efficiency.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import threading


@dataclass
class BulkCollectionConfig:
    """Centralized configuration for bulk collection operations"""
    
    # Date Range Settings
    default_period_days: int = 180
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Rate Limiting Settings
    delays = {
        'conservative': 2.0,
        'normal': 1.0,
        'aggressive': 0.5
    }
    
    # Batch Size Settings
    batch_sizes = {
        'small': 5,
        'medium': 10,
        'large': 20,
        'enterprise': 50
    }
    
    # Retry Settings
    max_retries: int = 3
    backoff_factor: float = 2.0
    
    # Parallel Processing Settings
    max_workers: int = 2
    worker_delay: float = 1.0
    
    def __post_init__(self):
        """Set default dates if not provided"""
        if self.end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        if self.start_date is None:
            start = datetime.now() - timedelta(days=self.default_period_days)
            self.start_date = start.strftime('%Y-%m-%d')
    
    def get_delay(self, level: str = 'normal') -> float:
        """Get delay for specified level"""
        return self.delays.get(level, self.delays['normal'])
    
    def get_batch_size(self, level: str = 'medium') -> int:
        """Get batch size for specified level"""
        return self.batch_sizes.get(level, self.batch_sizes['medium'])


class BulkCollectionError(Exception):
    """Custom exception for bulk collection operations"""
    pass


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"All {max_retries} attempts failed")
            
            raise BulkCollectionError(f"Failed after {max_retries} attempts: {last_exception}")
        return wrapper
    return decorator


class ProgressTracker:
    """Enhanced progress tracking with statistics"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.description = description
        self.start_time = time.time()
        self.completed = 0
        self.failed = 0
        self.pbar = tqdm(total=total_items, desc=description)
    
    def update(self, success: bool = True, count: int = 1):
        """Update progress with success/failure tracking"""
        if success:
            self.completed += count
        else:
            self.failed += count
        
        self.pbar.update(count)
        
        # Update progress bar postfix with statistics
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        success_rate = (self.completed / (self.completed + self.failed) * 100) if (self.completed + self.failed) > 0 else 0
        
        self.pbar.set_postfix({
            'Success': f"{success_rate:.1f}%",
            'Rate': f"{rate:.2f}/s"
        })
    
    def close(self):
        """Close progress bar and return final statistics"""
        self.pbar.close()
        total_time = time.time() - self.start_time
        
        return {
            'total_processed': self.completed + self.failed,
            'successful': self.completed,
            'failed': self.failed,
            'success_rate': self.completed / (self.completed + self.failed) if (self.completed + self.failed) > 0 else 0,
            'total_time': total_time,
            'processing_rate': self.completed / total_time if total_time > 0 else 0
        }


class ResultsManager:
    """Manages results display and saving"""
    
    @staticmethod
    def display_summary(data_dict: Dict[str, pd.DataFrame], title: str = "Results Summary"):
        """Display formatted summary of fetched data"""
        if not data_dict:
            print("❌ No data to summarize")
            return
        
        print(f"\n📊 **{title}**")
        print("=" * (len(title) + 8))
        
        # Basic statistics
        total_stocks = len(data_dict)
        total_records = sum(len(df) for df in data_dict.values() if df is not None)
        avg_records = total_records / total_stocks if total_stocks > 0 else 0
        
        print(f"📈 Stocks Fetched: {total_stocks}")
        print(f"📊 Total Records: {total_records:,}")
        print(f"⚖️ Average Records/Stock: {avg_records:.1f}")
        
        # Sample data info
        if data_dict:
            first_stock = list(data_dict.keys())[0]
            first_data = data_dict[first_stock]
            
            if first_data is not None and not first_data.empty:
                print(f"\n🔍 **Sample Data ({first_stock})**")
                print(f"   Records: {len(first_data)}")
                print(f"   Period: {first_data.index[0]} → {first_data.index[-1]}")
                print(f"   Columns: {list(first_data.columns)}")
    
    @staticmethod
    def save_with_metadata(data_dict: Dict[str, pd.DataFrame], 
                          metadata: Dict[str, Any], 
                          output_dir: Union[str, Path]):
        """Save data with comprehensive metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual stock files
        saved_files = []
        for symbol, data in data_dict.items():
            if data is not None and not data.empty:
                filename = f"{symbol.replace('.', '_')}.csv"
                filepath = output_path / filename
                data.to_csv(filepath)
                saved_files.append(str(filepath))
        
        # Save metadata
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return {
            'data_files': saved_files,
            'metadata_file': str(metadata_file),
            'total_files': len(saved_files)
        }


class BulkCollector:
    """Main bulk collection class with consolidated functionality"""
    
    def __init__(self, config: Optional[BulkCollectionConfig] = None):
        self.config = config or BulkCollectionConfig()
        self.logger = logging.getLogger(__name__)
    
    def fetch_sequential(self, 
                        stock_list: List[str], 
                        fetch_function: Callable,
                        level: str = 'medium') -> Dict[str, Any]:
        """Sequential fetching with progress tracking"""
        
        batch_size = self.config.get_batch_size(level)
        delay = self.config.get_delay(level)
        
        progress = ProgressTracker(len(stock_list), "Sequential Fetch")
        successful_data = {}
        failed_stocks = []
        
        try:
            # Process in batches
            for i in range(0, len(stock_list), batch_size):
                batch = stock_list[i:i + batch_size]
                
                try:
                    batch_data = fetch_function(
                        tickers=batch,
                        start_date=self.config.start_date,
                        end_date=self.config.end_date,
                        batch_size=len(batch),
                        delay_between_batches=delay
                    )
                    
                    successful_data.update(batch_data)
                    progress.update(success=True, count=len(batch))
                    
                except Exception as e:
                    self.logger.error(f"Batch failed: {e}")
                    failed_stocks.extend(batch)
                    progress.update(success=False, count=len(batch))
                
                # Rate limiting between batches
                if i + batch_size < len(stock_list):
                    time.sleep(delay)
        
        finally:
            stats = progress.close()
        
        return {
            'data': successful_data,
            'failed': failed_stocks,
            'statistics': stats,
            'metadata': {
                'fetch_method': 'sequential',
                'level': level,
                'config': self.config.__dict__,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def fetch_parallel(self, 
                      stock_list: List[str], 
                      fetch_function: Callable,
                      max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Parallel fetching with safety controls"""
        
        workers = max_workers or self.config.max_workers
        progress = ProgressTracker(len(stock_list), f"Parallel Fetch ({workers} workers)")
        
        successful_data = {}
        failed_stocks = []
        
        def fetch_single_safe(stock: str) -> tuple:
            """Safe single stock fetch with built-in delay"""
            try:
                time.sleep(self.config.worker_delay)
                result = fetch_function(
                    tickers=[stock],
                    start_date=self.config.start_date,
                    end_date=self.config.end_date
                )
                return stock, result.get(stock), None
            except Exception as e:
                return stock, None, str(e)
        
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_stock = {
                    executor.submit(fetch_single_safe, stock): stock 
                    for stock in stock_list
                }
                
                # Process results as they complete
                for future in as_completed(future_to_stock):
                    stock, data, error = future.result()
                    
                    if error is None and data is not None:
                        successful_data[stock] = data
                        progress.update(success=True)
                    else:
                        failed_stocks.append(stock)
                        progress.update(success=False)
                        if error:
                            self.logger.error(f"Failed to fetch {stock}: {error}")
        
        finally:
            stats = progress.close()
        
        return {
            'data': successful_data,
            'failed': failed_stocks,
            'statistics': stats,
            'metadata': {
                'fetch_method': 'parallel',
                'max_workers': workers,
                'config': self.config.__dict__,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def fetch_with_checkpoints(self, 
                              stock_list: List[str], 
                              fetch_function: Callable,
                              checkpoint_dir: Union[str, Path],
                              checkpoint_every: int = 100) -> Dict[str, Any]:
        """Fetch with checkpoint support for resume capability"""
        
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_path / f"checkpoint_{self.config.start_date}_{self.config.end_date}.json"
        
        # Load existing checkpoint
        completed_stocks = set()
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                completed_stocks = set(checkpoint_data.get('completed', []))
            print(f"📋 Resuming from checkpoint: {len(completed_stocks)} already completed")
        
        # Filter remaining stocks
        remaining_stocks = [s for s in stock_list if s not in completed_stocks]
        
        if not remaining_stocks:
            print("✅ All stocks already completed!")
            return {'data': {}, 'failed': [], 'statistics': {}, 'metadata': {}}
        
        print(f"🎯 Processing {len(remaining_stocks)} remaining stocks")
        
        # Fetch remaining stocks with regular checkpointing
        all_results = {'data': {}, 'failed': []}
        
        for i in range(0, len(remaining_stocks), checkpoint_every):
            batch = remaining_stocks[i:i + checkpoint_every]
            print(f"\n📦 Processing checkpoint batch {i//checkpoint_every + 1}")
            
            batch_results = self.fetch_sequential(batch, fetch_function)
            
            # Merge results
            all_results['data'].update(batch_results['data'])
            all_results['failed'].extend(batch_results['failed'])
            
            # Update checkpoint
            completed_stocks.update(batch_results['data'].keys())
            checkpoint_data = {
                'completed': list(completed_stocks),
                'last_update': datetime.now().isoformat(),
                'total_completed': len(completed_stocks),
                'total_target': len(stock_list)
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print(f"💾 Checkpoint saved: {len(completed_stocks)}/{len(stock_list)} completed")
        
        return all_results


# Convenience functions for different use cases
def create_beginner_collector() -> BulkCollector:
    """Create collector optimized for beginners"""
    config = BulkCollectionConfig()
    config.delays = {'normal': 2.0}  # Conservative delays
    config.batch_sizes = {'medium': 5}  # Small batches
    return BulkCollector(config)


def create_enterprise_collector() -> BulkCollector:
    """Create collector optimized for enterprise use"""
    config = BulkCollectionConfig()
    config.max_workers = 4
    config.batch_sizes = {'medium': 20}
    config.delays = {'normal': 0.5}
    return BulkCollector(config)


def quick_demo(stock_list: List[str], fetch_function: Callable, level: str = 'beginner') -> Dict[str, Any]:
    """Quick demonstration function"""
    if level == 'beginner':
        collector = create_beginner_collector()
        stock_list = stock_list[:5]  # Limit for demo
    else:
        collector = BulkCollector()
    
    print(f"🚀 Quick Demo: Fetching {len(stock_list)} stocks ({level} level)")
    
    results = collector.fetch_sequential(stock_list, fetch_function, level='small')
    
    # Display results
    ResultsManager.display_summary(results['data'], f"{level.title()} Demo Results")
    
    # Show statistics
    stats = results['statistics']
    print(f"\n📊 **Performance Statistics**")
    print(f"⏱️ Total Time: {stats['total_time']:.1f}s")
    print(f"📈 Success Rate: {stats['success_rate']:.1%}")
    print(f"🚀 Processing Rate: {stats['processing_rate']:.2f} stocks/sec")
    
    return results 