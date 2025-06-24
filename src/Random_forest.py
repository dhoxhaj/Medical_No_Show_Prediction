# MIT License
# Copyright (c) 2025 Debora Hoxhaj
# See LICENSE file for details.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from pathlib import Path

def load_data(path=None, target_column="No-show"):
    if path is None:
        base_dir = Path(__file__).parent.parent  
        path = base_dir / "Data" / "preprocessed_dataset.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def train_evaluate_rf(X_train, y_train, X_test, y_test, params):
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Set: {acc:.4f}")
    
    return model

def cross_validate_rf(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"\nCross-Validation Accuracy Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f}")
    return scores

def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, feature_names, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances[indices][:top_n], y=feature_names[indices][:top_n])
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_data()
    
    print("Class distribution in original dataset:")
    print(y.value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Uncomment if you want to balance with SMOTE
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    
    best_params = {
        'n_estimators': 250,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 5
    }
    
    rf_model = train_evaluate_rf(X_train, y_train, X_test, y_test, best_params)
    
    cross_validate_rf(rf_model, X, y)
    
    y_pred = rf_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importances(rf_model, X.columns)

if __name__ == "__main__":
    main()
