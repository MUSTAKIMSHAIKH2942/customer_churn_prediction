from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train_model(X_train, y_train):
    """Train a logistic regression model with scaling."""
    # Create a pipeline with scaling and logistic regression
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and save metrics to a file."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Ensure reports directory exists
    reports_dir = './reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Create a text file to save the metrics
    report_file_path = os.path.join(reports_dir, 'model_evaluation_report.txt')
    with open(report_file_path, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("-" * 80 + "\n")  # Separator between different models
    
    return accuracy, report

def save_model(model, filename):
    """Save the trained model to disk."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
