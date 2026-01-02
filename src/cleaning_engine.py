"""
Cleaning Engine: Automated data cleaning based on detected issues
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class CleaningEngine:
    """Automated data cleaning engine for energy consumption data."""
    
    def __init__(self, df: pd.DataFrame, consumption_col: str = 'consumption_kwh'):
        """
        Initialize cleaning engine.
        
        Args:
            df: DataFrame to clean
            consumption_col: Name of consumption column
        """
        self.df_original = df.copy()
        self.df = df.copy()
        self.consumption_col = consumption_col
        self.cleaning_log = []
        
        print(f"🧹 Cleaning Engine initialized")
        print(f"   Original data: {len(self.df_original)} rows")
    
    def clean_all(self, strategy: str = "conservative") -> pd.DataFrame:
        """
        Apply all cleaning operations.
        
        Args:
            strategy: "conservative" or "aggressive"
            
        Returns:
            Cleaned DataFrame
        """
        print(f"\n🧹 Starting automated cleaning ({strategy} strategy)...\n")
        
        # Order matters - clean in this sequence:
        self._remove_duplicates()
        self._fix_negative_values()
        self._fix_unit_errors()
        self._handle_outliers(strategy)
        self._handle_missing_values(strategy)
        self._handle_zero_values(strategy)
        
        print(f"\n✅ Cleaning complete!")
        print(f"   Final data: {len(self.df)} rows")
        print(f"   Rows removed: {len(self.df_original) - len(self.df)}")
        
        return self.df
    
    def _remove_duplicates(self):
        """Remove duplicate timestamps."""
        if 'timestamp' not in self.df.columns:
            return
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['timestamp'], keep='first')
        removed = initial_count - len(self.df)
        
        if removed > 0:
            self._log_action(f"Removed {removed} duplicate timestamps")
            print(f"  ✓ Removed {removed} duplicate timestamps")
    
    def _fix_negative_values(self):
        """Fix negative consumption values (always invalid)."""
        negative_mask = self.df[self.consumption_col] < 0
        count = negative_mask.sum()
        
        if count > 0:
            # Set to NaN (will be handled by missing value logic)
            self.df.loc[negative_mask, self.consumption_col] = np.nan
            self._log_action(f"Set {count} negative values to NaN")
            print(f"  ✓ Fixed {count} negative values (set to NaN)")
    
    def _fix_unit_errors(self):
        """Fix suspected unit conversion errors (values 1000x too large)."""
        # Values >500 kWh are likely Wh recorded as kWh
        unit_error_mask = self.df[self.consumption_col] > 500
        count = unit_error_mask.sum()
        
        if count > 0:
            # Divide by 1000 to convert Wh to kWh
            self.df.loc[unit_error_mask, self.consumption_col] /= 1000
            self._log_action(f"Fixed {count} unit conversion errors (divided by 1000)")
            print(f"  ✓ Fixed {count} suspected unit errors (Wh → kWh)")
    
    def _handle_outliers(self, strategy: str):
        """Handle statistical outliers."""
        # Calculate statistics on valid data only
        valid_data = self.df[
            (self.df[self.consumption_col].notna()) & 
            (self.df[self.consumption_col] >= 0)
        ][self.consumption_col]
        
        if len(valid_data) == 0:
            return
        
        mean = valid_data.mean()
        std = valid_data.std()
        threshold = 3.0
        
        # Identify outliers
        outlier_mask = (
            (self.df[self.consumption_col].notna()) & 
            (np.abs(self.df[self.consumption_col] - mean) > threshold * std)
        )
        count = outlier_mask.sum()
        
        if count > 0:
            if strategy == "aggressive":
                # Remove outliers
                self.df.loc[outlier_mask, self.consumption_col] = np.nan
                self._log_action(f"Set {count} outliers to NaN (aggressive)")
                print(f"  ✓ Removed {count} outliers (>3σ)")
            else:
                # Cap outliers at 3σ
                upper_bound = mean + threshold * std
                lower_bound = max(0, mean - threshold * std)
                
                self.df.loc[outlier_mask & (self.df[self.consumption_col] > upper_bound), 
                           self.consumption_col] = upper_bound
                self.df.loc[outlier_mask & (self.df[self.consumption_col] < lower_bound), 
                           self.consumption_col] = lower_bound
                
                self._log_action(f"Capped {count} outliers at ±3σ (conservative)")
                print(f"  ✓ Capped {count} outliers at ±3σ")
    
    def _handle_missing_values(self, strategy: str):
        """Handle missing values in consumption data."""
        missing_mask = self.df[self.consumption_col].isna()
        count = missing_mask.sum()
        
        if count > 0:
            if strategy == "aggressive":
                # Remove rows with missing values
                initial_len = len(self.df)
                self.df = self.df[~missing_mask]
                removed = initial_len - len(self.df)
                self._log_action(f"Removed {removed} rows with missing values")
                print(f"  ✓ Removed {removed} rows with missing values")
            else:
                # Forward fill (carry last known value forward)
                self.df[self.consumption_col] = self.df[self.consumption_col].fillna(method='ffill')
                
                # If still NaN at the start, use backward fill
                self.df[self.consumption_col] = self.df[self.consumption_col].fillna(method='bfill')
                
                self._log_action(f"Forward-filled {count} missing values")
                print(f"  ✓ Forward-filled {count} missing values")
    
    def _handle_zero_values(self, strategy: str):
        """Handle zero consumption values."""
        zero_mask = self.df[self.consumption_col] == 0
        count = zero_mask.sum()
        
        if count > 0:
            if strategy == "aggressive":
                # Treat zeros as missing and forward fill
                self.df.loc[zero_mask, self.consumption_col] = np.nan
                self.df[self.consumption_col] = self.df[self.consumption_col].fillna(method='ffill')
                self._log_action(f"Replaced {count} zero values with forward-fill")
                print(f"  ✓ Replaced {count} zeros with forward-fill")
            else:
                # Keep zeros (might be valid - building closed)
                self._log_action(f"Kept {count} zero values (potentially valid)")
                print(f"  ℹ️  Kept {count} zero values (may be valid)")
    
    def get_cleaning_summary(self) -> str:
        """Generate summary of cleaning operations."""
        summary = "\n" + "="*60 + "\n"
        summary += "  CLEANING SUMMARY\n"
        summary += "="*60 + "\n\n"
        
        summary += f"Original rows: {len(self.df_original):,}\n"
        summary += f"Final rows: {len(self.df):,}\n"
        summary += f"Rows removed: {len(self.df_original) - len(self.df):,}\n\n"
        
        summary += "Actions taken:\n"
        for i, action in enumerate(self.cleaning_log, 1):
            summary += f"  {i}. {action}\n"
        
        summary += "\n" + "="*60 + "\n"
        
        return summary
    
    def get_before_after_stats(self) -> Dict:
        """Compare statistics before and after cleaning."""
        stats_before = {
            'count': len(self.df_original),
            'mean': self.df_original[self.consumption_col].mean(),
            'median': self.df_original[self.consumption_col].median(),
            'std': self.df_original[self.consumption_col].std(),
            'min': self.df_original[self.consumption_col].min(),
            'max': self.df_original[self.consumption_col].max(),
            'missing': self.df_original[self.consumption_col].isna().sum(),
        }
        
        stats_after = {
            'count': len(self.df),
            'mean': self.df[self.consumption_col].mean(),
            'median': self.df[self.consumption_col].median(),
            'std': self.df[self.consumption_col].std(),
            'min': self.df[self.consumption_col].min(),
            'max': self.df[self.consumption_col].max(),
            'missing': self.df[self.consumption_col].isna().sum(),
        }
        
        return {
            'before': stats_before,
            'after': stats_after
        }
    
    def _log_action(self, action: str):
        """Log a cleaning action."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.cleaning_log.append(f"[{timestamp}] {action}")