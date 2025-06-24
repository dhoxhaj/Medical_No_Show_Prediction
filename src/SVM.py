# MIT License
# Copyright (c) 2025 Debora Hoxhaj
# See LICENSE file for details.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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

def preprocess_and_split(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    return X_train_bal, X_test, y_train_bal, y_test, scaler


def train_evaluate_svm(X_train, X_test, y_train, y_test):
    svm_clf = LinearSVC(
        C=0.01,
        loss='squared_hinge',
        penalty='l2',
        max_iter=10000,
        class_weight='balanced',
        dual=True,
        random_state=42
    )
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    cv_scores = cross_val_score(svm_clf, np.vstack((X_train, X_test)), 
                                np.concatenate((y_train, y_test)), cv=3)
    print("Mean CV Accuracy:", cv_scores.mean())

    return svm_clf, y_pred, y_test, cv_scores


def plot_results(y_test, y_pred, cv_scores):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Show', 'Show'], yticklabels=['Not Show', 'Show'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).iloc[:-1, :].T

    plt.figure(figsize=(8, 6))
    sns.heatmap(class_report_df, annot=True, cmap='coolwarm', cbar=False, fmt='.2f')
    plt.title('Classification Report Heatmap')
    plt.tight_layout()
    plt.show()

    cv_score_series = pd.Series(cv_scores, index=[f'Fold {i+1}' for i in range(len(cv_scores))])
    plt.figure(figsize=(8, 6))
    sns.barplot(x=cv_score_series.index, y=cv_score_series.values, palette='viridis')
    plt.ylim(0, 1)
    plt.xlabel('Cross-Validation Fold')
    plt.ylabel('Accuracy Score')
    plt.title('Cross-Validation Scores')
    plt.tight_layout()
    plt.show()


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(X, y)
    svm_model, y_pred, y_test, cv_scores = train_evaluate_svm(X_train, X_test, y_train, y_test)
    plot_results(y_test, y_pred, cv_scores)


if __name__ == "__main__":
    main()
