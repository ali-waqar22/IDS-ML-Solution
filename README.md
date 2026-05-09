# A1-IS-IDS-ML-Solution: AI-Powered Intrusion Detection

## Project Title
AI-Powered Intrusion Detection Solution with Real-time Threat Detection

## Objective
This project fulfills Assignment 1 for Information Security. The objective is to design, develop, and evaluate a proof-of-concept Machine Learning model that automatically classifies network traffic as normal or malicious. This acts as an intelligent augmentation to traditional NIDS (Network Intrusion Detection Systems).

## Scenario
As a security analyst at "SecureNet Corp", this solution provides a proactive line of defense against modern network attacks such as DDoS, Brute Force, Web Attacks, and Infiltrations.

## Dataset Setup Instructions
This project uses a synthetic generator modeled after the real-world **CIC-IDS2017** dataset.
1. Run the Streamlit Dashboard.
2. Go to the **Dashboard** tab.
3. Under "Quick Start Options", choose "Generate Synthetic CICIDS2017 Dataset" and click **Generate Dataset**.
4. The dataset includes realistic flow features (Duration, Packet lengths, Flags) and 5 target classes: BENIGN, DDoS, Brute_Force, Web_Attack, and Infiltration.

*(Alternatively, you can upload a custom CSV dataset using the upload widget).*

## How to Run the Code

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Installation
Navigate to the project directory and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Dashboard
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your web browser.

### 4. Running the Jupyter Notebook
For a step-by-step code walkthrough, open the provided notebook:
```bash
jupyter notebook CLO4_IDS_Workflow.ipynb
```

## Brief Summary of Results
- **Preprocessing:** Handled missing values, scaled numeric features using Standard/MinMax Scalers, and encoded categorical variables.
- **Models Implemented:** k-Nearest Neighbors, Decision Tree, Random Forest, and Naive Bayes.
- **Performance:** Random Forest typically achieves the highest accuracy (>95%) and best recall for minority attack classes, making it highly effective for minimizing false negatives in a security context.
- **Evaluation:** The solution provides detailed Confusion Matrices, ROC curves, and Feature Importance metrics to justify the model's decisions to security management.

## Author
Ali Waqar
GitHub: [ali-waqar22](https://github.com/ali-waqar22)
Course: Information Security

