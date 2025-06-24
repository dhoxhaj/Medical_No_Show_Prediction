# Medical Appointment No-Show Prediction
## 1. Introduction
Missed medical appointments, or no-shows, cause significant inefficiencies in healthcare systems worldwide, leading to wasted resources and delayed care. In Brazil's public healthcare system, addressing no-shows is critical to improve service delivery and access.
This project uses machine learning models to predict patient no-shows based on demographics, medical history, and appointment details, enabling proactive interventions like reminders or rescheduling.
We work with the Medical Appointment No-Shows dataset from Kaggle, containing over 110,000 records. The target is imbalanced: approximately 20% no-shows.

## 2. Feature Overview

Original Features

Demographics: Age, Gender, Neighborhood
Health Info: Hypertension, Diabetes, Alcoholism, Handicap, Scholarship
Appointment Details: Scheduled day, Appointment day, SMS reminder
Target: No-show (0 = attended, 1 = no-show)

Engineered Features

DaysBetween: Days between scheduling and appointment
AppointmentWeekday: Weekday of appointment
ScheduledWeekday: Weekday of scheduling
IsWeekendAppointment: Boolean if appointment on weekend

## 3. Data Preprocessing

Removed irrelevant columns (PatientId, AppointmentID, ScheduledDay).
Transformed target to binary (NoShow: 1 = no-show, 0 = attended).
Corrected invalid data (e.g., negative ages).
One-hot encoded categorical variables.
Scaled numerical features using StandardScaler.
Balanced training set with SMOTE after train-test split to avoid leakage.

## 4. Modeling Pipeline
The project focuses on four main machine learning models implemented in the /src folder:

4.1 Feedforward Neural Network (FFNN)

Multiple architectures tested (2, 10, 15, 30, 40, 50 hidden layers).
Batch normalization, dropout, and different activation functions explored.
Early stopping used to avoid overfitting.

4.2 Random Forest

Tuned parameters: number of trees, max depth, min samples split/leaf.
Balanced training set with SMOTE.
Feature importance visualized.

4.3 Stochastic Gradient Descent Classifier (SGD)

Linear model with hinge loss and L1 penalty.
Scaled features.
Balanced dataset with SMOTE.

4.4 Support Vector Machine (SVM)

LinearSVC with balanced class weights.
Regularization tuned via hyperparameter C.
Feature scaling applied.
Cross-validation for performance validation.

## 5. Evaluation

Confusion matrices and classification reports for detailed class-wise performance.
Metrics: Accuracy, Precision, Recall, F1-score.
Cross-validation used to estimate model generalization.
Visualization of confusion matrices and metric bar plots included.

## 6. Software and Tools

Python 3.12
VS Code for development and testing.
Libraries: pandas, numpy, scikit-learn, imbalanced-learn, tensorflow, matplotlib, seaborn
Google Colab used during initial experimentation for scalability and GPU support.

## 7. Usage

Place your dataset preprocessed_dataset.csv inside the /Data folder. Each model script in /src reads this file, performs preprocessing, trains the model, evaluates, and outputs performance metrics and plots.
To run any model, execute the corresponding Python script in the /src folder

## 8. Acknowledgements & References

Dataset from Kaggle: Medical Appointment No-Shows
Inspired by various machine learning studies on no-show prediction, including Random Forest and ensemble methods.
