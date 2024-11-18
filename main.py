import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model_training import train_model, evaluate_model, save_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    # Step 1: Load and preprocess the data
    data = load_data('./data/raw/Data_file.csv')
    processed_data, numerical_cols = preprocess_data(data)
    
    # Step 2: Split the data into training and test sets
    target_column = 'Churn'
    X_train, X_test, y_train, y_test = split_data(processed_data, target_column)
    
    # Step 3: Define models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    best_report = None
    
    # Step 4: Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Step 5: Evaluate the model
        accuracy, report = evaluate_model(model, X_test, y_test, model_name)
        print(f"{model_name} Accuracy: {accuracy}")
        
        # Track the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name
            best_report = report
    
    # Step 6: Print the best model's performance
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy}")
    print("Classification Report:")
    print(best_report)
    
    # Step 7: Save the best model
    save_model(best_model, 'models/final_model.pkl')
    print(f"Best model ({best_model_name}) saved!")

if __name__ == "__main__":
    main()
