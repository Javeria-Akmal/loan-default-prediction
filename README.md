Loan Default Prediction

End-to-end ML/DL pipeline for predicting loan default risk, deployed as a live Streamlit app.

🔗 Live App: https://loan-default-prediction-gqhurms2yxako7rav8ah2x.streamlit.app

Problem Statement

Predict whether a borrower will default on a loan, based on financial and demographic features — helping lenders assess risk before approval.

Dataset

Loan Default Dataset (Kaggle) — ~255,000 records with borrower financial and loan attributes.

Approach


Preprocessing: Handled class imbalance using SMOTE
Machine Learning models: Logistic Regression, Random Forest, XGBoost
Deep Learning models: ANN, DNN, CNN, LSTM
Deployment: Backend + Frontend built and deployed as a Streamlit web app


Results

ModelTypeAccuracyXGBoostBest ML model~82%CNNBest DL model~84.7%

Tech Stack

Python · Scikit-learn · XGBoost · TensorFlow · Streamlit

Project Structure

├── Backend/       # Model training & prediction logic
├── Frontend/       # Streamlit app interface
└── runtime.txt      # Deployment environment config
