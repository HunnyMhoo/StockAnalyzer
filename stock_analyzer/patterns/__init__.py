"""
Patterns subpackage for pattern detection, labeling, and scanning.
"""

from .labeler import (
    PatternLabel,
    PatternLabeler,
    LabelValidator,
    save_labeled_patterns,
    load_labeled_patterns,
    is_false_breakout,
    ValidationError,
    PatternLabelError,
)

from .scanner import (
    PatternScanner,
    ScanningConfig,
    scan_hk_stocks_for_patterns,
    PatternScanningError,
)

__all__ = [
    # Pattern labeling
    "PatternLabel",
    "PatternLabeler",
    "LabelValidator", 
    "save_labeled_patterns",
    "load_labeled_patterns",
    "is_false_breakout",
    "ValidationError",
    "PatternLabelError",
    
    # Pattern scanning
    "PatternScanner",
    "ScanningConfig",
    "scan_hk_stocks_for_patterns",
    "PatternScanningError",
] 