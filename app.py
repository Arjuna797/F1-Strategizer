from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import random
import traceback

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

def load_model():
    try:
        with open('saved_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("❌ Model file 'saved_model.pkl' not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

MODEL_DATA = load_model()

if MODEL_DATA:
    model = MODEL_DATA['model']
    scaler = MODEL_DATA['scaler']
    imputer = MODEL_DATA['imputer']
    features = MODEL_DATA['features']
    racer_names = MODEL_DATA['racer_names']
    AvgLapTime = MODEL_DATA.get('AvgLapTime', None)
    
    print("✅ Model loaded successfully!")
    print("📊 Model keys:", MODEL_DATA.keys())
    print("🏎️ Racers loaded:", racer_names)
    print("⏱️ AvgLapTime:", AvgLapTime)
else:
    print("❌ No model loaded - using demo mode")
    model = scaler = imputer = racer_names = None
    features = []
    AvgLapTime = None

# Driver-team mapping (2024 F1 Season)
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

# Base driver performance variations (simulate different driver skills)
DRIVER_VARIATIONS = {
    'VER': -0.3, 'PER': -0.1, 'NOR': -0.2, 'PIA': -0.15, 
    'LEC': -0.05, 'SAI': 0.05, 'RUS': 0.0, 'HAM': 0.1,
    'ALO': 0.15, 'STR': 0.25, 'HUL': 0.3, 'MAG': 0.35,
    'OCO': 0.2, 'GAS': 0.3, 'ALB': 0.4, 'SAR': 0.45,
    'BOT': 0.25, 'ZHO': 0.35, 'TSU': 0.4, 'RIC': 0.2
}

# Demo racers if no model is loaded
DEMO_RACERS = ['VER', 'NOR', 'LEC', 'RUS', 'SAI', 'HAM', 'PIA', 'ALO']

def show_graph():
    
    df = pd.read_json('my_dataframe.json')
    
    print("IN GRAPH function")    

    # 9. Plot feature importances
    plt.figure(figsize=(8, 5))
    importances = model.feature_importances_
    plt.barh(features, importances, color="skyblue")
    plt.title("Feature Importance in Race Time Prediction")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
    
    
    # 10. Plot Clean-Air Pace vs. Predicted Lap Time
    final = df.sort_values("PredictedLapTime (s)").reset_index(drop=True)
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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("🚀 Prediction request received")
        
        # Get request data
        data = request.get_json(force=True)
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data received in request'
            }), 400
            
        print("📊 Request data:", data)
        
        # Check required keys
        required_keys = [
            'qualifying_time', 'rain_probability', 'temperature',
            'team_performance', 'clean_air_pace', 'position_change', 'sector_time'
        ]
        
        missing = [k for k in required_keys if k not in data]
        if missing:
            return jsonify({
                'success': False,
                'error': f"Missing required fields: {', '.join(missing)}"
            }), 400

        # Use demo mode if no model is available
        if not MODEL_DATA:
            print("⚠️ No model loaded - using demo mode")
            return generate_demo_prediction(data)

        # Map incoming JSON to DataFrame with correct column names
        base_input = {
            'QualifyingTime (s)': float(data['qualifying_time']),
            'RainProbability': float(data['rain_probability']) / 100.0,
            'Temperature (C)': float(data['temperature']),
            'TeamPerformanceScore': float(data['team_performance']),
            'CleanAirRacePace (s)': float(data['clean_air_pace']),
            'AveragePositionChange': float(data['position_change']),
            'TotalSectorTime (s)': float(data['sector_time'])
        }
        
        print("🔄 Base input created:", base_input)

        # Generate predictions for all drivers
        all_predictions = []
        
        current_racers = racer_names if racer_names else DEMO_RACERS
        print(f"🏎️ Predicting for drivers: {current_racers}")
        
        for driver in current_racers:
            try:
                # Create driver-specific variations
                driver_input = base_input.copy()
                variation = DRIVER_VARIATIONS.get(driver, 0.0)
                
                # Apply driver-specific adjustments
                driver_input['QualifyingTime (s)'] += variation
                driver_input['CleanAirRacePace (s)'] += variation
                driver_input['TotalSectorTime (s)'] += variation
                
                # Add some randomness for realistic variation
                random_factor = random.uniform(-0.1, 0.1)
                driver_input['QualifyingTime (s)'] += random_factor
                
                # Create DataFrame and make prediction
                input_df = pd.DataFrame([driver_input], columns=features)
                X_imp = imputer.transform(input_df)
                X_scaled = scaler.transform(X_imp)
                predicted_time = model.predict(X_scaled)[0]
                
                # Calculate confidence based on driver performance and conditions
                base_confidence = 85.0
                if driver in ['VER', 'NOR', 'LEC', 'PIA']: # Top drivers
                    base_confidence = 90.0
                elif driver in ['RUS', 'HAM', 'SAI']: # Mid-tier
                    base_confidence = 87.0
                
                # Add variation based on conditions
                if data['rain_probability'] > 50:
                    base_confidence -= 3.0
                if data['team_performance'] > 0.7:
                    base_confidence += 2.0
                
                confidence = base_confidence + random.uniform(-2.0, 2.0)
                confidence = max(82.0, min(95.0, confidence))
                
                all_predictions.append({
                    'driver': driver,
                    'team': DRIVER_TEAMS.get(driver, 'Unknown Team'),
                    'predicted_time': float(predicted_time),
                    'confidence': round(confidence, 1)
                })
                
            except Exception as e:
                print(f"❌ Error predicting for driver {driver}: {e}")
                continue

        if not all_predictions:
            return jsonify({
                'success': False,
                'error': 'Failed to generate any predictions'
            }), 500

        # Sort by predicted time to get podium
        all_predictions.sort(key=lambda x: x['predicted_time'])
        podium = all_predictions[:3]

        # Get the primary prediction (fastest lap time)
        primary_time = podium[0]['predicted_time']
        
        print(f"✅ Predicted Time: {primary_time}")
        print(f"🏆 Podium: {[(p['driver'], p['predicted_time']) for p in podium]}")
        
        show_graph()
        
        return jsonify({
            'success': True,
            'predicted_lap_time': round(primary_time, 3),
            'confidence': podium[0]['confidence'],
            'podium': podium,
            'all_predictions': all_predictions[:8]  # Top 8 for extended view
        })

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in predict endpoint: {error_msg}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {error_msg}'
        }), 500

def generate_demo_prediction(data):
    """Generate demo predictions when no ML model is available"""
    print("🎭 Generating demo predictions...")
    
    # Base lap time around Monaco GP typical times
    base_time = 79.0 + random.uniform(-2.0, 2.0)
    
    # Generate predictions for demo drivers
    all_predictions = []
    for i, driver in enumerate(DEMO_RACERS):
        # Add variation based on driver skill and request parameters
        time_variation = DRIVER_VARIATIONS.get(driver, 0.0)
        time_variation += (float(data.get('rain_probability', 20)) / 100.0) * random.uniform(-0.5, 0.5)
        time_variation += (1.0 - float(data.get('team_performance', 0.7))) * random.uniform(0, 1.0)
        
        predicted_time = base_time + time_variation + random.uniform(-0.3, 0.3)
        confidence = 85.0 + random.uniform(-5.0, 10.0)
        confidence = max(80.0, min(95.0, confidence))
        
        all_predictions.append({
            'driver': driver,
            'team': DRIVER_TEAMS.get(driver, 'Demo Team'),
            'predicted_time': float(predicted_time),
            'confidence': round(confidence, 1)
        })
    
    # Sort by predicted time
    all_predictions.sort(key=lambda x: x['predicted_time'])
    podium = all_predictions[:3]
    primary_time = podium[0]['predicted_time']
    
    print(f"🎭 Demo Predicted Time: {primary_time}")
    print(f"🎭 Demo Podium: {[(p['driver'], p['predicted_time']) for p in podium]}")
    
    return jsonify({
        'success': True,
        'predicted_lap_time': round(primary_time, 3),
        'confidence': podium[0]['confidence'],
        'podium': podium,
        'all_predictions': all_predictions,
        'demo_mode': True
    })

@app.route('/health', methods=['GET'])
def health():
    model_status = "loaded" if MODEL_DATA else "not_loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status,
        "drivers_count": len(racer_names) if racer_names else len(DEMO_RACERS)
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "features": features if features else [],
        "drivers": racer_names if racer_names else DEMO_RACERS,
        "model_loaded": MODEL_DATA is not None
    })

# Add CORS headers for development
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    print("🏎️ Starting F1 Monaco GP Predictor...")
    print(f"📊 Model Status: {'✅ Loaded' if MODEL_DATA else '❌ Not Loaded (Demo Mode)'}")
    app.run(host='0.0.0.0', port=5000, debug=True)
