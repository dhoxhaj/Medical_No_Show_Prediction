# MIT License
# Copyright (c) 2025 Debora Hoxhaj
# See LICENSE file for details.

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


from pathlib import Path

def load_data(path=None, target_column="No-show"):
    if path is None:
        base_dir = Path(__file__).parent.parent  
        path = base_dir / "Data" / "preprocessed_dataset.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def preprocess_and_split(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    return X_train_bal, X_test, y_train_bal, y_test, scaler


def train_evaluate_sgd(X_train, X_test, y_train, y_test):
    model = SGDClassifier(
        loss='hinge',
        alpha=0.001,
        penalty='l1',
        learning_rate='optimal',
        max_iter=1000,
        tol=1e-3,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.3f}")
    
    return model, y_pred, (acc, prec, rec, f1)


def save_results(metrics, filename="experiment_results.xlsx", run=1):
    new_result = pd.DataFrame([{
        "Run": run,
        "Model": "SGDClassifier",
        "Accuracy": metrics[0],
        "Precision": metrics[1],
        "Recall": metrics[2],
        "F1 Score": metrics[3]
    }])
    
    if os.path.exists(filename):
        existing = pd.read_excel(filename)
        final_df = pd.concat([existing, new_result], ignore_index=True)
    else:
        final_df = new_result
        
    final_df.to_excel(filename, index=False)


def plot_results(y_test, y_pred, metrics):
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('SGDClassifier - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    sns.barplot(x=metric_names, y=list(metrics), palette='pastel')
    plt.ylim(0, 1)
    plt.title('SGDClassifier - Performance Metrics')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.show()


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(X, y)
    model, y_pred, metrics = train_evaluate_sgd(X_train, X_test, y_train, y_test)
    save_results(metrics)
    plot_results(y_test, y_pred, metrics)


if __name__ == "__main__":
    main()
