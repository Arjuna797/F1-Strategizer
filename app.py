from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np
import pandas as pd
import random
import traceback
import sqlite3
import datetime
import os
import json

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent tkinter issues
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import os

app = Flask(__name__)

# Load secret keys and configurations from environment variables
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default_dev_secret_key')  # replace in prod
# Add further secret keys or API keys as needed
# e.g. app.config['EXAMPLE_API_KEY'] = os.environ.get('EXAMPLE_API_KEY')

# Note: It's important to keep these secrets out of source control by using environment variables.

# Ensure static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Database setup
DATABASE = 'f1.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                qualifying_time REAL,
                rain_probability REAL,
                temperature REAL,
                team_performance REAL,
                clean_air_pace REAL,
                position_change REAL,
                sector_time REAL,
                predicted_lap_time REAL,
                confidence REAL,
                podium TEXT,
                all_predictions TEXT
            )
        ''')
        conn.commit()

init_db()

# Load model
def load_model():
    try:
        with open('saved_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print("❌ Model load error:", e)
        return None

MODEL_DATA = load_model()

if MODEL_DATA:
    model = MODEL_DATA['model']
    scaler = MODEL_DATA['scaler']
    imputer = MODEL_DATA['imputer']
    features = MODEL_DATA['features']
    racer_names = MODEL_DATA['racer_names']
else:
    model = scaler = imputer = None
    features = []
    racer_names = []

# Driver mapping
DRIVER_TEAMS = {
    'VER': 'Red Bull Racing', 'PER': 'Red Bull Racing',
    'NOR': 'McLaren', 'PIA': 'McLaren',
    'RUS': 'Mercedes', 'HAM': 'Mercedes',
    'SAI': 'Ferrari', 'LEC': 'Ferrari',
    'STR': 'Aston Martin', 'ALO': 'Aston Martin',
    'HUL': 'Haas F1 Team', 'MAG': 'Haas F1 Team',
    'OCO': 'Alpine', 'GAS': 'Alpine',
    'ALB': 'Williams', 'SAR': 'Williams',
    'BOT': 'Kick Sauber', 'ZHO': 'Kick Sauber',
    'TSU': 'RB F1 Team', 'RIC': 'RB F1 Team'
}

# More realistic driver variations
DRIVER_VARIATIONS = {
    'VER': -0.8, 'NOR': -0.5, 'LEC': -0.45, 'SAI': -0.40,
    'HAM': -0.35, 'RUS': -0.32, 'PIA': -0.30, 'ALO': -0.25,
    'PER': -0.22,
    'STR': +0.15, 'OCO': +0.18, 'GAS': +0.22, 'RIC': +0.25,
    'TSU': +0.28, 'ALB': +0.30,
    'HUL': +0.45, 'MAG': +0.50, 'BOT': +0.55, 'ZHO': +0.60,
    'SAR': +0.70
}

DEMO_RACERS = ['VER','NOR','LEC','RUS','SAI','HAM','PIA','ALO']

# ⭐ NEW — Save graph instead of showing graph
def show_graph(predictions=None):
    try:
        df = pd.read_json("my_dataframe.json")

        # Filter out rows with null values for trend line calculation
        df_clean = df.dropna(subset=["QualifyingTime (s)", "PredictedLapTime (s)"])

        # Save Lap Time Distribution Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df["PredictedLapTime (s)"], bins=20, alpha=0.7, color="#ffd700", edgecolor="#e10600")
        plt.axvline(df["PredictedLapTime (s)"].mean(), color="#e10600", linestyle='--', linewidth=2, label=f'Mean: {df["PredictedLapTime (s)"].mean():.3f}s')
        plt.title("Distribution of Predicted Lap Times", fontsize=14, fontweight='bold')
        plt.xlabel("Predicted Lap Time (seconds)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("static/feature_importance.png")
        plt.close()

        # Save Qualifying Time vs Predicted Lap Time Scatter Plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df["QualifyingTime (s)"], df["PredictedLapTime (s)"],
                            c=df["TeamPerformanceScore"], cmap='viridis', s=80, alpha=0.8, edgecolors='black')

        # Add trend line with error handling
        try:
            if len(df_clean) >= 2:  # Need at least 2 points for linear fit
                z = np.polyfit(df_clean["QualifyingTime (s)"], df_clean["PredictedLapTime (s)"], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df_clean["QualifyingTime (s)"].min(), df_clean["QualifyingTime (s)"].max(), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')
            else:
                print("⚠ Not enough data points for trend line")
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"⚠ Trend line calculation failed: {e}")
            # Add a simple reference line instead
            plt.axline((df["QualifyingTime (s)"].mean(), df["PredictedLapTime (s)"].mean()),
                      slope=0, color='r', linestyle='--', alpha=0.8, linewidth=2, label='Mean Reference')

        plt.colorbar(scatter, label='Team Performance Score')
        plt.title("Qualifying Time vs Predicted Lap Time", fontsize=14, fontweight='bold')
        plt.xlabel("Qualifying Time (seconds)")
        plt.ylabel("Predicted Lap Time (seconds)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("static/pace_graph.png")
        plt.close()

        # Save Driver Performance Bar Chart if predictions are provided
        if predictions:
            plt.figure(figsize=(12, 6))
            drivers = [p['driver'] for p in predictions[:8]]
            times = [p['predicted_time'] for p in predictions[:8]]
            colors = ['#e10600' if i < 3 else '#ffd700' for i in range(len(drivers))]  # Red for podium, gold for others
            plt.bar(drivers, times, color=colors, alpha=0.8, edgecolor='black')
            plt.title("Predicted Lap Times for Top 8 Drivers", fontsize=14, fontweight='bold')
            plt.xlabel("Driver")
            plt.ylabel("Predicted Lap Time (seconds)")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig("static/driver_performance.png")
            plt.close()

        print("📊 New graphs saved successfully in /static")

    except Exception as e:
        print("⚠ Graph creation failed:", e)


@app.route('/')
def index():
    firebase_api_key = os.environ.get('FIREBASE_API_KEY', 'your_default_firebase_api_key_here')
    return render_template("index.html",
                           feature_graph=url_for('static', filename='feature_importance.png'),
                           pace_graph=url_for('static', filename='pace_graph.png'),
                           driver_graph=url_for('static', filename='driver_performance.png'),
                           firebase_api_key=firebase_api_key
                           )

@app.route('/predictor')
def predictor():
    return render_template("predictor.html",
                           feature_graph=url_for('static', filename='feature_importance.png'),
                           pace_graph=url_for('static', filename='pace_graph.png'),
                           driver_graph=url_for('static', filename='driver_performance.png')
                           )


@app.route('/predict', methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        required = ['qualifying_time','rain_probability','temperature','team_performance',
                    'clean_air_pace','position_change','sector_time']

        for k in required:
            if k not in data:
                return jsonify(success=False, error=f"Missing {k}")

        if not MODEL_DATA:
            return generate_demo(data)

        base_input = {
            'QualifyingTime (s)': float(data['qualifying_time']),
            'RainProbability': float(data['rain_probability']) / 100.0,
            'Temperature (C)': float(data['temperature']),
            'TeamPerformanceScore': float(data['team_performance']),
            'CleanAirRacePace (s)': float(data['clean_air_pace']),
            'AveragePositionChange': float(data['position_change']),
            'TotalSectorTime (s)': float(data['sector_time'])
        }

        predictions = []
        drivers = racer_names if racer_names else DEMO_RACERS

        for driver in drivers:
            d_input = base_input.copy()
            variation = DRIVER_VARIATIONS.get(driver, 0)

            for f in ['QualifyingTime (s)','CleanAirRacePace (s)','TotalSectorTime (s)']:
                d_input[f] += variation

            d_input['QualifyingTime (s)'] += random.uniform(-0.35, 0.35)

            df = pd.DataFrame([d_input], columns=features)
            X_imp = imputer.transform(df)
            X_scaled = scaler.transform(X_imp)
            predicted_time = float(model.predict(X_scaled)[0])

            if data['rain_probability'] > 50:
                if driver in ['HAM','ALO','NOR']:
                    predicted_time -= 0.25
                if driver in ['STR','SAR','ZHO']:
                    predicted_time += 0.40

            conf = 85 + random.uniform(-3,3)

            predictions.append({
                'driver': driver,
                'team': DRIVER_TEAMS.get(driver,"Team"),
                'predicted_time': predicted_time,
                'confidence': round(conf,1)
            })

        predictions.sort(key=lambda x: x['predicted_time'])
        podium = predictions[:3]

        # ⭐ Save prediction to database
        with get_db() as conn:
            conn.execute('''
                INSERT INTO predictions (
                    timestamp, qualifying_time, rain_probability, temperature,
                    team_performance, clean_air_pace, position_change, sector_time,
                    predicted_lap_time, confidence, podium, all_predictions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.datetime.now().isoformat(),
                data['qualifying_time'],
                data['rain_probability'],
                data['temperature'],
                data['team_performance'],
                data['clean_air_pace'],
                data['position_change'],
                data['sector_time'],
                round(podium[0]['predicted_time'], 3),
                podium[0]['confidence'],
                json.dumps(podium),
                json.dumps(predictions[:8])
            ))
            conn.commit()

        # ⭐ Create graphs after each prediction request
        show_graph(predictions=predictions[:8])

        return jsonify(success=True,
                       predicted_lap_time=round(podium[0]['predicted_time'],3),
                       confidence=podium[0]['confidence'],
                       podium=podium,
                       all_predictions=predictions[:8],
                       graph_feature=url_for('static', filename='feature_importance.png'),
                       graph_pace=url_for('static', filename='pace_graph.png'),
                       graph_driver=url_for('static', filename='driver_performance.png')
                       )

    except Exception as e:
        traceback.print_exc()
        return jsonify(success=False, error=str(e))


# DEMO MODE
def generate_demo(data):
    base_time = 79 + random.uniform(-2, 2)
    preds = []

    for d in DEMO_RACERS:
        v = DRIVER_VARIATIONS.get(d, 0)
        t = base_time + v + random.uniform(-0.5, 0.5)

        preds.append({
            "driver": d,
            "team": DRIVER_TEAMS.get(d),
            "predicted_time": t,
            "confidence": round(85 + random.uniform(-5, 10), 1)
        })

    preds.sort(key=lambda x: x['predicted_time'])
    podium = preds[:3]

    # ⭐ Save demo prediction to database
    with get_db() as conn:
        conn.execute('''
            INSERT INTO predictions (
                timestamp, qualifying_time, rain_probability, temperature,
                team_performance, clean_air_pace, position_change, sector_time,
                predicted_lap_time, confidence, podium, all_predictions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now().isoformat(),
            data.get('qualifying_time', 0),
            data.get('rain_probability', 0),
            data.get('temperature', 0),
            data.get('team_performance', 0),
            data.get('clean_air_pace', 0),
            data.get('position_change', 0),
            data.get('sector_time', 0),
            round(podium[0]['predicted_time'], 3),
            podium[0]['confidence'],
            json.dumps(podium),
            json.dumps(preds)
        ))
        conn.commit()

    show_graph(predictions=preds)

    return jsonify(success=True, demo_mode=True,
                   podium=podium,
                   all_predictions=preds,
                   graph_feature="static/feature_importance.png",
                   graph_pace="static/pace_graph.png",
                   graph_driver="static/driver_performance.png"
                   )


@app.route('/health')
def health():
    return jsonify(status="healthy", model_loaded=MODEL_DATA is not None)


if __name__ == "__main__":
    print("🚀 Server running...")
    # Disable debug mode if FLASK_ENV is set to 'production'
    flask_env = os.environ.get('FLASK_ENV', 'development')
    debug_mode = False if flask_env == 'production' else True
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
