# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Define the dataset path
DATASET_PATH = 'synthetic_training_data.csv'

# Check if the dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: '{DATASET_PATH}' not found.")
    print("Please run 'generate_synthetic_data.py' first to create the dataset.")
else:
    # 1. Load your dataset
    data = pd.read_csv(DATASET_PATH)

    # 2. Define features (X) and target (y)
    features = ['lrs_score', 'prs_score']
    target = 'has_diabetes'
    X = data[features]
    y = data[target]

    # 3. Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # [cite_start]4. Initialize and train the Random Forest model (as mentioned in your report) [cite: 83]
    print("🚀 Training the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate the model
    predictions = model.predict(X_test)
    print(f"📊 Model Accuracy on test data: {accuracy_score(y_test, predictions):.2f}")

    # 6. Save the trained model to a file
    joblib.dump(model, 't2d_risk_model.joblib')
    print("✅ Model saved to 't2d_risk_model.joblib'. Your backend is ready to use it!")