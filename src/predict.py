# src/predict.py

import pickle
import pandas as pd

def load_model(model_file):
    """Load the trained model from a file."""
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, X_new):
    """Predict churn for new data with scaling applied."""
    # Predict churn for new data, assuming the model pipeline includes scaling
    return model.predict(X_new)
