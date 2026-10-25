import numpy as np
import pandas as pd

np.random.seed(42)

dates = pd.date_range(start="2016-01-01", end="2025-12-01", freq="MS")
n = len(dates)
month = dates.month

wet_season = np.isin(month, [6, 7, 8, 9, 10, 11]).astype(float)
rainfall = 80 + wet_season * 180 + np.random.normal(0, 25, n)
rainfall = np.clip(rainfall, 10, None)


temp_avg = 27 + 2.5 * np.sin((month - 3) / 12 * 2 * np.pi) + np.random.normal(0, 0.6, n)
temp_min = np.round(temp_avg - np.random.uniform(2.5, 4.0, n), 2)
temp_max = np.round(temp_avg + np.random.uniform(2.5, 4.0, n), 2)

humidity = 65 + wet_season * 15 + np.random.normal(0, 4, n)
humidity = np.round(np.clip(humidity, 40, 100), 2)

trend = np.linspace(0, 15, n)
seasonal = 10 * np.sin((month - 2) / 12 * 2 * np.pi)
water_supply = 250 + trend + seasonal + 0.05 * (rainfall - 150) + np.random.normal(0, 8, n)
water_supply = np.round(np.clip(water_supply, 150, None), 2)

df = pd.DataFrame({
    "date": dates.strftime("%Y-%m-%d"),
    "water_supply": water_supply,
    "rainfall": np.round(rainfall, 2),
    "temp_min": temp_min,
    "temp_max": temp_max,
    "humidity": humidity,
})

missing_idx = np.random.choice(n, size=3, replace=False)
df.loc[missing_idx, "water_supply"] = np.nan

df.to_csv("data/sample_data.csv", index=False)
print(f"Wrote data/sample_data.csv with {len(df)} rows")
