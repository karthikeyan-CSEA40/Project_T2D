# app.py (Focused on Random Forest Prediction)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
import json
import joblib
import numpy as np

# Initialize the Flask web application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# --- CONFIGURATION ---
PLINK_PATH = './plink.exe'
WEIGHTS_FILE_PATH = './weights.txt'
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 't2d_risk_model.joblib'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- LOAD THE TRAINED MACHINE LEARNING MODEL ---
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Machine Learning model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at '{MODEL_PATH}'.")
    print("Please run 'train_model.py' to create the model file.")
    model = None


def calculate_lrs(lrs_data):
    """Calculates the total Lifestyle Risk Score (LRS)."""
    return sum(lrs_data.values())

def run_plink_prs(vcf_filepath):
    """Executes PLINK to calculate the Polygenic Risk Score (PRS)."""
    output_prefix = os.path.join(UPLOAD_FOLDER, 'prs_result')
    command = [
        PLINK_PATH,
        '--vcf', vcf_filepath,
        '--score', WEIGHTS_FILE_PATH, '1', '2', '3', 'header', 'sum',
        '--out', output_prefix
    ]
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        profile_path = output_prefix + '.profile'
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                f.readline()
                data_line = f.readline()
                if data_line:
                    return float(data_line.split()[-1])
        return None
    except Exception as e:
        print(f"An unexpected error occurred during PLINK execution: {e}")
        return None


def run_ml_prediction(lrs_score, prs_score):
    """Uses the pre-trained model to predict diabetes risk probability."""
    if model is None:
        raise RuntimeError("Machine learning model is not loaded.")
    
    input_features = np.array([[lrs_score, prs_score]])
    prediction_probability = model.predict_proba(input_features)[0][1]
    return prediction_probability

def stratify_risk(probability):
    """Categorizes the prediction probability into risk levels."""
    if probability <= 0.33:
        return "Low Risk"
    elif probability <= 0.66:
        return "Moderate Risk"
    else:
        return "High Risk"

@app.route('/predict', methods=['POST'])
def predict():
    """Main API endpoint to orchestrate the prediction process."""
    if model is None:
        return jsonify({'error': 'Server is not ready; ML model not loaded.'}), 503

    vcf_file = request.files.get('vcf_file')
    if not vcf_file or vcf_file.filename == '':
        return jsonify({'error': 'No VCF file provided or file is empty'}), 400
        
    lrs_data = json.loads(request.form.get('lrs_data', '{}'))
    
    vcf_filename = os.path.join(UPLOAD_FOLDER, vcf_file.filename)
    vcf_file.save(vcf_filename)

    prs_score = None 
    try:
        lrs_score = calculate_lrs(lrs_data)
        prs_score = run_plink_prs(vcf_filename)
        
        if prs_score is None:
            return jsonify({'error': 'Failed to calculate PRS. Check server logs.'}), 500

        ml_prediction = run_ml_prediction(lrs_score, prs_score)
        risk_category = stratify_risk(ml_prediction)

        return jsonify({
            'lrs_score': lrs_score,
            'prs_score': prs_score,
            'ml_prediction': ml_prediction,
            'risk_category': risk_category,
        })
    finally:
        # This block ensures cleanup happens even if other parts fail
        try:
            if os.path.exists(vcf_filename):
                os.remove(vcf_filename)
            for ext in ['.bed', '.bim', '.fam', '.log', '.nosex', '.profile']:
                filepath = os.path.join(UPLOAD_FOLDER, 'prs_result' + ext)
                if os.path.exists(filepath):
                    os.remove(filepath)
        except OSError as e:
            print(f"Warning during file cleanup: {e}")


if __name__ == '__main__':
    app.run(debug=True, port=5000)

