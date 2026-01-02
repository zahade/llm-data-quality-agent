"""
Data Analyzer: Statistical anomaly detection for energy consumption data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


class DataAnalyzer:
    """Analyzes energy data for quality issues using statistical methods."""
    
    def __init__(self, df: pd.DataFrame, consumption_column: str = 'consumption_kwh'):
        """
        Initialize analyzer with data.
        
        Args:
            df: DataFrame with energy data
            consumption_column: Name of consumption column
        """
        self.df = df.copy()
        self.consumption_col = consumption_column
        self.issues = {}
        
    def detect_all_issues(self) -> Dict[str, any]:
        """
        Run all anomaly detection methods.
        
        Returns:
            Dictionary containing all detected issues
        """
        print("🔍 Running anomaly detection...")
        
        self.issues = {
            'missing_values': self._detect_missing_values(),
            'negative_values': self._detect_negative_values(),
            'zero_values': self._detect_zero_values(),
            'outliers': self._detect_outliers(),
            'duplicates': self._detect_duplicates(),
            'unit_errors': self._detect_unit_errors(),
        }
        
        # Get detailed information
        self.issues['missing_indices'] = self.df[self.df[self.consumption_col].isna()].index.tolist()
        self.issues['negative_indices'] = self.df[self.df[self.consumption_col] < 0].index.tolist()
        self.issues['zero_indices'] = self.df[self.df[self.consumption_col] == 0].index.tolist()
        
        # Outlier detection with indices
        outlier_info = self._get_outlier_details()
        self.issues['outlier_indices'] = outlier_info['indices']
        self.issues['outlier_values'] = outlier_info['values']
        
        print(f"✅ Detection complete. Found {self._count_total_issues()} total issues.")
        
        return self.issues
    
    def _detect_missing_values(self) -> int:
        """Detect missing values in consumption column."""
        missing = self.df[self.consumption_col].isna().sum()
        if missing > 0:
            print(f"  ❌ Missing values: {missing}")
        return int(missing)
    
    def _detect_negative_values(self) -> int:
        """Detect negative consumption values (invalid)."""
        negative = (self.df[self.consumption_col] < 0).sum()
        if negative > 0:
            print(f"  ❌ Negative values: {negative}")
        return int(negative)
    
    def _detect_zero_values(self) -> int:
        """Detect zero consumption values (may be valid or invalid)."""
        zeros = (self.df[self.consumption_col] == 0).sum()
        if zeros > 0:
            print(f"  ⚠️  Zero values: {zeros}")
        return int(zeros)
    
    def _detect_outliers(self, threshold: float = 3.0) -> int:
        """
        Detect outliers using z-score method.
        
        Args:
            threshold: Number of standard deviations for outlier
            
        Returns:
            Count of outliers
        """
        # Calculate z-scores (ignore NaN and negative values)
        valid_data = self.df[
            (self.df[self.consumption_col].notna()) & 
            (self.df[self.consumption_col] >= 0)
        ][self.consumption_col]
        
        if len(valid_data) == 0:
            return 0
            
        z_scores = np.abs(stats.zscore(valid_data))
        outliers = (z_scores > threshold).sum()
        
        if outliers > 0:
            print(f"  ❌ Outliers (>{threshold}σ): {outliers}")
        
        return int(outliers)
    
    def _get_outlier_details(self, threshold: float = 3.0) -> Dict:
        """Get detailed information about outliers."""
        valid_mask = (self.df[self.consumption_col].notna()) & (self.df[self.consumption_col] >= 0)
        valid_data = self.df[valid_mask][self.consumption_col]
        
        if len(valid_data) == 0:
            return {'indices': [], 'values': []}
        
        mean = valid_data.mean()
        std = valid_data.std()
        
        # Find outliers
        outlier_mask = valid_mask & (
            np.abs(self.df[self.consumption_col] - mean) > threshold * std
        )
        
        outlier_indices = self.df[outlier_mask].index.tolist()
        outlier_values = self.df[outlier_mask][self.consumption_col].tolist()
        
        return {
            'indices': outlier_indices,
            'values': outlier_values,
            'mean': mean,
            'std': std
        }
    
    def _detect_duplicates(self) -> int:
        """Detect duplicate timestamps."""
        if 'timestamp' not in self.df.columns:
            return 0
            
        duplicates = self.df.duplicated(subset=['timestamp']).sum()
        
        if duplicates > 0:
            print(f"  ❌ Duplicate timestamps: {duplicates}")
        
        return int(duplicates)
    
    def _detect_unit_errors(self) -> int:
        """
        Detect potential unit conversion errors (values 1000x too large).
        Common error: Wh recorded as kWh.
        """
        valid_data = self.df[
            (self.df[self.consumption_col].notna()) & 
            (self.df[self.consumption_col] >= 0)
        ][self.consumption_col]
        
        if len(valid_data) == 0:
            return 0
        
        # Typical residential: 0-50 kWh/hour
        # Values >500 kWh might be Wh recorded as kWh
        suspected_errors = (valid_data > 500).sum()
        
        if suspected_errors > 0:
            print(f"  ❌ Suspected unit errors (>500 kWh): {suspected_errors}")
        
        return int(suspected_errors)
    
    def _count_total_issues(self) -> int:
        """Count total number of issues detected."""
        return sum([
            self.issues.get('missing_values', 0),
            self.issues.get('negative_values', 0),
            self.issues.get('zero_values', 0),
            self.issues.get('outliers', 0),
            self.issues.get('duplicates', 0),
            self.issues.get('unit_errors', 0),
        ])
    
    def get_sample_issues(self, n: int = 5) -> Dict[str, List]:
        """
        Get sample of each issue type for LLM analysis.
        
        Args:
            n: Number of samples per issue type
            
        Returns:
            Dictionary with sample data for each issue type
        """
        samples = {}
        
        # Sample missing values
        if self.issues.get('missing_values', 0) > 0:
            missing_idx = self.issues['missing_indices'][:n]
            samples['missing'] = self.df.iloc[missing_idx].to_dict('records')
        
        # Sample negative values
        if self.issues.get('negative_values', 0) > 0:
            negative_idx = self.issues['negative_indices'][:n]
            samples['negative'] = self.df.iloc[negative_idx].to_dict('records')
        
        # Sample outliers
        if self.issues.get('outliers', 0) > 0:
            outlier_idx = self.issues['outlier_indices'][:n]
            samples['outliers'] = self.df.iloc[outlier_idx].to_dict('records')
        
        return samples
    
    def generate_analysis_report(self) -> str:
        """Generate a detailed text report of findings."""
        report = "📊 DATA QUALITY ANALYSIS REPORT\n"
        report += "="*60 + "\n\n"
        
        report += f"Total Records: {len(self.df):,}\n"
        report += f"Total Issues: {self._count_total_issues():,}\n\n"
        
        report += "Issue Breakdown:\n"
        for issue_type, count in self.issues.items():
            if isinstance(count, int) and count > 0:
                pct = (count / len(self.df)) * 100
                report += f"  - {issue_type.replace('_', ' ').title()}: {count:,} ({pct:.2f}%)\n"
        
        return report