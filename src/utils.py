"""
Utility functions for the LLM Data Quality Agent
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_energy_data(file_path: str) -> pd.DataFrame:
    """
    Load energy consumption data from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with energy data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime if exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        print(f"✅ Loaded {len(df)} rows from {file_path}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise


def get_basic_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Calculate basic statistics for a column.
    
    Args:
        df: DataFrame
        column: Column name to analyze
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'count': int(df[column].count()),
        'missing': int(df[column].isna().sum()),
        'mean': float(df[column].mean()) if df[column].notna().any() else 0,
        'median': float(df[column].median()) if df[column].notna().any() else 0,
        'std': float(df[column].std()) if df[column].notna().any() else 0,
        'min': float(df[column].min()) if df[column].notna().any() else 0,
        'max': float(df[column].max()) if df[column].notna().any() else 0,
        'q25': float(df[column].quantile(0.25)) if df[column].notna().any() else 0,
        'q75': float(df[column].quantile(0.75)) if df[column].notna().any() else 0,
    }
    
    return stats


def format_stats_for_llm(stats: Dict[str, Any]) -> str:
    """
    Format statistics in a readable way for LLM input.
    
    Args:
        stats: Dictionary of statistics
        
    Returns:
        Formatted string
    """
    return f"""
Data Statistics:
- Total records: {stats['count']:,}
- Missing values: {stats['missing']:,} ({stats['missing']/stats['count']*100:.1f}%)
- Mean: {stats['mean']:.2f}
- Median: {stats['median']:.2f}
- Std Dev: {stats['std']:.2f}
- Min: {stats['min']:.2f}
- Max: {stats['max']:.2f}
- Q1 (25%): {stats['q25']:.2f}
- Q3 (75%): {stats['q75']:.2f}
"""


def create_issue_summary(issues: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of data quality issues.
    
    Args:
        issues: Dictionary of detected issues
        
    Returns:
        Formatted summary string
    """
    summary = "🔍 Data Quality Issues Detected:\n\n"
    
    total_issues = 0
    
    if issues.get('missing_values', 0) > 0:
        summary += f"❌ Missing Values: {issues['missing_values']:,}\n"
        total_issues += issues['missing_values']
        
    if issues.get('negative_values', 0) > 0:
        summary += f"❌ Negative Values: {issues['negative_values']:,}\n"
        total_issues += issues['negative_values']
        
    if issues.get('zero_values', 0) > 0:
        summary += f"⚠️  Zero Values: {issues['zero_values']:,}\n"
        total_issues += issues['zero_values']
        
    if issues.get('outliers', 0) > 0:
        summary += f"❌ Outliers (>3σ): {issues['outliers']:,}\n"
        total_issues += issues['outliers']
        
    if issues.get('duplicates', 0) > 0:
        summary += f"❌ Duplicate Timestamps: {issues['duplicates']:,}\n"
        total_issues += issues['duplicates']
        
    if issues.get('unit_errors', 0) > 0:
        summary += f"❌ Suspected Unit Errors: {issues['unit_errors']:,}\n"
        total_issues += issues['unit_errors']
        
    summary += f"\n📊 Total Issues: {total_issues:,}\n"
    
    return summary


def save_cleaned_data(df: pd.DataFrame, original_path: str) -> str:
    """
    Save cleaned data to processed folder.
    
    Args:
        df: Cleaned DataFrame
        original_path: Original file path
        
    Returns:
        Path to saved file
    """
    # Create output filename
    base_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(base_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{name_without_ext}_cleaned_{timestamp}.csv"
    
    # Create output path
    output_path = os.path.join("data", "processed", output_name)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved cleaned data to: {output_path}")
    
    return output_path


def get_api_key() -> str:
    """
    Get Groq API key from environment.
    
    Returns:
        API key string
    """
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file. "
            "Please add your API key to the .env file."
        )
    
    return api_key


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")