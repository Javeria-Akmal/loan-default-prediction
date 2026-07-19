<h1>Loan Default Prediction</h1>
<b>End-to-end ML/DL pipeline</b> for predicting loan default risk, deployed as a live <b>Streamlit app</b>.

🔗 <b>Live App:</b> https://loan-default-prediction-gqhurms2yxako7rav8ah2x.streamlit.app

<h2>Problem Statement</h2>
Predict whether a borrower will <b>default on a loan</b>, based on financial and demographic features — helping lenders assess risk before approval.
<h2>Dataset</h2>
<a href="https://www.kaggle.com/datasets/nikhil1e9/loan-default"><b>Loan Default Dataset</b></a> (Kaggle) — <b>~255,000 records</b> with borrower financial and loan attributes.
<h2>Approach</h2>
<ul>
<li><b>Preprocessing:</b> Handled class imbalance using <b>SMOTE</b></li>
<li><b>Machine Learning models:</b> Logistic Regression, Random Forest, <b>XGBoost</b></li>
<li><b>Deep Learning models:</b> ANN, DNN, <b>CNN</b>, LSTM</li>
<li><b>Deployment:</b> Backend + Frontend built and deployed as a <b>Streamlit web app</b></li>
</ul>
<h2>Results</h2>
<table>
<tr><th>Model</th><th>Type</th><th>Accuracy</th></tr>
<tr><td><b>XGBoost</b></td><td>Best ML model</td><td><b>~82%</b></td></tr>
<tr><td><b>CNN</b></td><td>Best DL model</td><td><b>~84.7%</b></td></tr>
</table>
<h2>Tech Stack</h2>
<code>Python</code> · <code>Scikit-learn</code> · <code>XGBoost</code> · <code>TensorFlow</code> · <code>Streamlit</code>
<h2>Project Structure</h2>
<pre>
├── Backend/       # Model training & prediction logic
├── Frontend/       # Streamlit app interface
└── runtime.txt      # Deployment environment config

