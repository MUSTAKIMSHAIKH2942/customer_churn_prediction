# src/feature_engineering.py

from sklearn.preprocessing import PolynomialFeatures

def generate_polynomial_features(X):
    """Generate polynomial features for interaction terms (optional)."""
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    return X_poly
