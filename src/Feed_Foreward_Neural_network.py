# MIT License
# Copyright (c) 2025 Debora Hoxhaj
# See LICENSE file for details.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from pathlib import Path

def load_data(path=None, target_column="No-show"):
    if path is None:
        base_dir = Path(__file__).parent.parent  
        path = base_dir / "Data" / "preprocessed_dataset.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_bal, y_test

def build_model(input_dim, n_hidden_layers=50, dropout_rate=0.3):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    
    for _ in range(n_hidden_layers - 1):  # already have 1 layer above
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    model = build_model(input_dim=X_train.shape[1], n_hidden_layers=50, dropout_rate=0.3)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Accuracy on Test Set: {:.4f}".format(accuracy_score(y_test, y_pred)))

if __name__ == "__main__":
    train_and_evaluate()
