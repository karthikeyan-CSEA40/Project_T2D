# generate_synthetic_data.py

import pandas as pd
import numpy as np

# --- Configuration ---
NUM_SAMPLES = 2000  # Create a dataset with 2000 people
LRS_MIN, LRS_MAX = 5, 25  # Realistic range for a lifestyle score
PRS_MEAN, PRS_STD = 0.0, 0.1  # Realistic range for a polygenic score

# --- Generate Features ---
# Create a random Lifestyle Risk Score (LRS) for each person
lrs_scores = np.random.uniform(LRS_MIN, LRS_MAX, NUM_SAMPLES)

# Create a random Polygenic Risk Score (PRS) for each person
prs_scores = np.random.normal(PRS_MEAN, PRS_STD, NUM_SAMPLES)

# --- Generate Target (The "has_diabetes" outcome) ---
# We create a logical rule: higher scores increase diabetes probability.
# We'll normalize the scores to combine their effects.
risk_metric = (lrs_scores / LRS_MAX) + (prs_scores / (PRS_MEAN + 3 * PRS_STD))

# Use a sigmoid function to create a probability based on the risk metric
probabilities = 1 / (1 + np.exp(-risk_metric * 2))

# Determine the final outcome based on this probability
has_diabetes = (probabilities > np.random.rand(NUM_SAMPLES)).astype(int)

# --- Create and Save the DataFrame ---
synthetic_df = pd.DataFrame({
    'lrs_score': lrs_scores,
    'prs_score': prs_scores,
    'has_diabetes': has_diabetes
})

synthetic_df.to_csv('synthetic_training_data.csv', index=False)

print(f"✅ Successfully generated 'synthetic_training_data.csv' with {NUM_SAMPLES} samples.")
print("You can now use this file to train your model.")