import fastf1
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

# 1. Load and preprocess 2024 Monaco race session
session = fastf1.get_session(2024, 8, "R")
session.load()
# After filtering non-null laps, immediately copy
laps = session.laps.dropna(subset=["LapTime","Sector1Time","Sector2Time","Sector3Time"]).copy()

# Now adding new columns won’t warn
for col in ["LapTime","Sector1Time","Sector2Time","Sector3Time"]:
    laps[f"{col} (s)"] = laps[col].dt.total_seconds()

# Aggregate mean sector times per driver
sector_times = (
    laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
    .mean()
    .reset_index()
)
sector_times["TotalSectorTime (s)"] = (
    sector_times["Sector1Time (s)"]
    + sector_times["Sector2Time (s)"]
    + sector_times["Sector3Time (s)"]
)

# 2. Qualifying 2025 Monaco GP data and clean-air race pace
clean_air = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667,
    "NOR": 93.428600, "ALO": 94.784333, "PIA": 93.232111,
    "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250,
    "HUL": 95.345455, "OCO": 95.682128
}

qualifying = pd.DataFrame({
    "Driver": ["VER","NOR","PIA","RUS","SAI","ALB","LEC","OCO","HAM","STR","GAS","ALO","HUL"],
    "QualifyingTime (s)": [
        70.669, 69.954, 70.129, np.nan,
        71.362, 71.213, 70.063, 70.942,
        70.382, 72.563, 71.994, 70.924, 71.596
    ]
})
qualifying["CleanAirRacePace (s)"] = qualifying["Driver"].map(clean_air)

# Map drivers to teams and normalize team points
team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131,
    "Williams": 51, "Ferrari": 114, "Haas": 20,
    "Aston Martin": 14, "Kick Sauber": 6,
    "Racing Bulls": 10, "Alpine": 7
}
max_pts = max(team_points.values())
team_score = {t: p/max_pts for t, p in team_points.items()}

driver_team = {
    "VER":"Red Bull","NOR":"McLaren","PIA":"McLaren","LEC":"Ferrari",
    "RUS":"Mercedes","HAM":"Mercedes","GAS":"Alpine","ALO":"Aston Martin",
    "STR":"Aston Martin","HUL":"Kick Sauber","OCO":"Alpine","SAI":"Ferrari","ALB":"Williams"
}
qualifying["Team"] = qualifying["Driver"].map(driver_team)
qualifying["TeamPerformanceScore"] = qualifying["Team"].map(team_score)

# Average position change at Monaco GP
avg_pos_change = {
    "VER": -1.0, "NOR": 1.0, "PIA": 0.2, "RUS": 0.5,
    "SAI": -0.3, "ALB": 0.8, "LEC": -1.5, "OCO": -0.2,
    "HAM": 0.3, "STR": 1.1, "GAS": -0.4, "ALO": -0.6, "HUL": 0.0
}
qualifying["AveragePositionChange"] = qualifying["Driver"].map(avg_pos_change)

# 3. Fetch weather data via WeatherAPI.com
import os

api_key = os.environ.get('WEATHER_API_KEY', 'your_default_weather_api_key_here')
lat, lon = 43.7384, 7.4246
weather_url = (
    f"http://api.weatherapi.com/v1/forecast.json"
    f"?key={api_key}&q={lat},{lon}&days=3&aqi=no&alerts=no"
)
wd = requests.get(weather_url).json()

# Extract forecast for 2025-05-25 13:00 local time
forecast_data = None
for day in wd.get("forecast", {}).get("forecastday", []):
    for hr in day.get("hour", []):
        if hr["time"] == "2025-05-25 13:00":
            forecast_data = hr
            break
    if forecast_data:
        break

rain_prob = forecast_data.get("chance_of_rain", 0) / 100 if forecast_data else 0
temp_c    = forecast_data.get("temp_c", 20)        if forecast_data else 20

# 4. Merge datasets
# Filter to drivers present in all sources
common = (
    set(qualifying.Driver)
    & set(sector_times.Driver)
    & set(clean_air.keys())
)
qualifying = qualifying[qualifying.Driver.isin(common)]
sector_times = sector_times[sector_times.Driver.isin(common)]

# Compute average lap time per driver
avg_lap = (
    laps.groupby("Driver")["LapTime (s)"]
    .mean()
    .reset_index()
    .rename(columns={"LapTime (s)": "AvgLapTime"})
)

df = (
    qualifying.merge(sector_times[["Driver", "TotalSectorTime (s)"]], on="Driver")
             .merge(avg_lap, on="Driver")
)
df["RainProbability"] = rain_prob
df["Temperature (C)"] = temp_c

# 5. Features and target
features = [
    "QualifyingTime (s)", "RainProbability", "Temperature (C)",
    "TeamPerformanceScore", "CleanAirRacePace (s)",
    "AveragePositionChange", "TotalSectorTime (s)"
]
X = df[features]
y = df["AvgLapTime"]

# 6. Imputation, scaling, train-test split
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X)

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_imp)

X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.3, random_state=37
)
   
# print("IN X_TEST",X_test)   
with open("X_test", 'wb') as file:
    pickle.dump(X_test, file)

# 7. Train Gradient Boosting Regressor
model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.7, max_depth=3, random_state=37
)
model.fit(X_train, y_train)

filename = "saved.pkl"
with open(filename, 'wb') as file:
    pickle.dump(model, file)
   
# with open("y_test", 'wb') as file:
#     pickle.dump(y_test, file)  


# with open("df",'wb') as file:
#     pickle.dump(df,file)

# 8. Predictions and evaluation
df["PredictedLapTime (s)"] = model.predict(scaler.transform(imp.transform(X)))
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Model Mean Absolute Error (MAE): {mae:.2f} seconds\n")

# Predicted finishing order
df.to_json('my_dataframe.json')
final = df.sort_values("PredictedLapTime (s)").reset_index(drop=True)
print("🏁 Predicted 2025 Monaco GP Order 🏁")
print(final[["Driver", "PredictedLapTime (s)"]])

# 9. Plot feature importances
plt.figure(figsize=(8, 5))
importances = model.feature_importances_
plt.barh(features, importances, color="skyblue")
plt.title("Feature Importance in Race Time Prediction")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 10. Plot Clean-Air Pace vs. Predicted Lap Time
plt.figure(figsize=(10, 6))
plt.scatter(final["CleanAirRacePace (s)"], final["PredictedLapTime (s)"], s=60)
for idx, driver in final.iterrows():
    plt.annotate(driver["Driver"],
                 (driver["CleanAirRacePace (s)"], driver["PredictedLapTime (s)"]),
                 xytext=(5, 4), textcoords="offset points")
plt.title("Effect of Clean-Air Race Pace on Predicted Lap Time")
plt.xlabel("Clean-Air Race Pace (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.tight_layout()
plt.show()
