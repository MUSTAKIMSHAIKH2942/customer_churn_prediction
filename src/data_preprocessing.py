# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load raw data from CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the customer churn dataset."""
    # Drop irrelevant columns like customerID (since it's not needed for modeling)
    df = df.drop(columns=['customerID'])
    
    # Handle missing values:
    # Replace missing 'TotalCharges' with the mean of the column
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    
    # For categorical columns, replace missing values with the most frequent value (mode)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encode categorical variables using Label Encoding (convert strings to numerical values)
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Scaling numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])  # Apply standardization
    
    return df, numerical_cols

def split_data(df, target_column):
    """Split data into features (X) and target variable (y), and split into train/test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
