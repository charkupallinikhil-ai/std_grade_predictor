import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

def main():
    print("Loading dataset...")
    try:
        df = pd.read_excel('student_dataset.xlsx')
    except FileNotFoundError:
        print("Error: student_dataset.xlsx not found.")
        return

    # Clean the dataset
    print("Preprocessing data...")
    # 'StudyHours' contains 'unknown' strings, replace with NaN
    df['StudyHours'] = pd.to_numeric(df['StudyHours'], errors='coerce')
    
    # We will predict 'Grade' using 'Age', 'StudyHours', 'Attendance', 'Assignments', 'Midterm'
    features = ['Age', 'StudyHours', 'Attendance', 'Assignments', 'Midterm']
    target = 'Grade'
    
    # Check if target is present
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found in the dataset.")
        return
        
    X = df[features]
    y = df[target]

    # Preprocessing pipeline
    # We impute missing values with median
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features)
        ])

    # Model pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

    print("Training model...")
    # Train the model
    model.fit(X, y)
    print("Model trained successfully.")

    # Save the pipeline
    joblib.dump(model, 'model.pkl')
    print("Model saved to 'model.pkl'.")

if __name__ == "__main__":
    main()
