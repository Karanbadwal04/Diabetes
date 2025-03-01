from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the trained diabetes model and scaler
diabetes_model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        features = np.array([[
            int(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetes_pedigree']),
            int(request.form['age'])
        ]])

        # Scale input data
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = diabetes_model.predict(scaled_features)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)