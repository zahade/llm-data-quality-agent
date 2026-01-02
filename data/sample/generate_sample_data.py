import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("🔄 Generating sample energy data...")

np.random.seed(42)

# Generate 1 year of hourly data
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(hours=i) for i in range(8760)]  # 365 days * 24 hours

# Base consumption pattern
hours = np.arange(8760)
daily_pattern = 50 + 30 * np.sin(2 * np.pi * hours / 24)  # Daily cycle
seasonal_pattern = 20 * np.sin(2 * np.pi * hours / (24 * 365))  # Seasonal
noise = np.random.normal(0, 5, 8760)
consumption = daily_pattern + seasonal_pattern + noise + 100

# Create DataFrame
data = pd.DataFrame({
    'timestamp': dates,
    'consumption_kwh': consumption,
    'temperature_c': 15 + 10 * np.sin(2 * np.pi * hours / (24 * 365)) + np.random.normal(0, 3, 8760),
    'building_id': np.random.choice(['A', 'B', 'C'], 8760),
})

print(f"📊 Created {len(data)} rows of clean data")

# INTENTIONALLY INTRODUCE QUALITY ISSUES:

# 1. Missing values (5% random)
missing_indices = np.random.choice(8760, size=int(8760 * 0.05), replace=False)
data.loc[missing_indices, 'consumption_kwh'] = np.nan
print(f"❌ Added {len(missing_indices)} missing values")

# 2. Negative values (impossible - equipment error)
negative_indices = np.random.choice(8760, size=20, replace=False)
data.loc[negative_indices, 'consumption_kwh'] = -np.random.uniform(10, 50, 20)
print(f"❌ Added 20 negative values (invalid)")

# 3. Extreme outliers (10x normal - meter malfunction)
outlier_indices = np.random.choice(8760, size=30, replace=False)
data.loc[outlier_indices, 'consumption_kwh'] = data['consumption_kwh'].mean() * np.random.uniform(8, 12, 30)
print(f"❌ Added 30 extreme outliers")

# 4. Duplicate timestamps
duplicate_rows = data.iloc[100:105].copy()
data = pd.concat([data, duplicate_rows], ignore_index=True)
print(f"❌ Added {len(duplicate_rows)} duplicate rows")

# 5. Unit inconsistencies (Wh instead of kWh - 1000x error)
unit_error_indices = np.random.choice(len(data), size=50, replace=False)
data.loc[unit_error_indices, 'consumption_kwh'] = data.loc[unit_error_indices, 'consumption_kwh'] * 1000
print(f"❌ Added 50 unit conversion errors (1000x)")

# 6. Zero values (meter offline)
zero_indices = np.random.choice(len(data), size=15, replace=False)
data.loc[zero_indices, 'consumption_kwh'] = 0
print(f"❌ Added 15 zero values")

# Save to CSV
output_path = 'energy_data_with_issues.csv'
data.to_csv(output_path, index=False)

print(f"\n✅ Generated sample data: {output_path}")
print(f"📈 Total rows: {len(data)}")
print(f"📊 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
print("\nData Quality Summary:")
print(f"  - Missing values: {data['consumption_kwh'].isna().sum()}")
print(f"  - Negative values: {(data['consumption_kwh'] < 0).sum()}")
print(f"  - Zero values: {(data['consumption_kwh'] == 0).sum()}")
print(f"  - Suspected outliers (>500 kWh): {(data['consumption_kwh'] > 500).sum()}")
print(f"  - Duplicate timestamps: {data.duplicated(subset=['timestamp']).sum()}")