"""
model_manager.py — ML Model Training and Evaluation
===================================================
Handles training, evaluation, and saving/loading of ML models.
Supports k-NN, Decision Tree, Random Forest, and Naive Bayes.
"""

import time
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def get_model(model_name, params=None):
    """Factory function to instantiate models based on name and params."""
    if params is None:
        params = {}
        
    if model_name == 'k-NN':
        k = params.get('k', 5)
        metric = params.get('metric', 'minkowski')
        # Map Streamlit UI names to sklearn names
        if metric == 'Euclidean': metric = 'euclidean'
        elif metric == 'Manhattan': metric = 'manhattan'
        elif metric == 'Cosine': metric = 'cosine'
        
        return KNeighborsClassifier(n_neighbors=k, metric=metric)
        
    elif model_name == 'Decision Tree':
        max_depth = params.get('max_depth', None)
        criterion = params.get('criterion', 'gini').lower()
        return DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
        
    elif model_name == 'Random Forest':
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', None)
        criterion = params.get('criterion', 'gini').lower()
        return RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            criterion=criterion, 
            random_state=42,
            n_jobs=-1
        )
        
    elif model_name == 'Naive Bayes':
        var_smoothing = params.get('var_smoothing', 1e-9)
        return GaussianNB(var_smoothing=float(var_smoothing))
        
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(model, X_train, y_train, X_test, y_test, class_names=None):
    """
    Train a model and evaluate its performance.
    
    Returns a dictionary of metrics and evaluation artifacts.
    """
    results = {}
    
    # Training phase
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediction phase
    start_time = time.time()
    y_pred = model.predict(X_test)
    
    # Get probabilities if the model supports it
    try:
        y_prob = model.predict_proba(X_test)
        results['y_prob'] = y_prob
    except AttributeError:
        results['y_prob'] = None
        
    pred_time = time.time() - start_time
    
    # Calculate metrics
    results['accuracy'] = accuracy_score(y_test, y_pred)
    
    # Use macro average for multiclass precision/recall/f1
    results['precision'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
    results['recall'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
    results['f1'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Detailed per-class metrics
    results['classification_report'] = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    results['classification_report_str'] = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    # Feature importance if available (Tree-based models)
    if hasattr(model, 'feature_importances_'):
        results['feature_importances'] = model.feature_importances_
    else:
        results['feature_importances'] = None
        
    results['model'] = model
    results['train_time'] = train_time
    results['pred_time'] = pred_time
    results['y_pred'] = y_pred
    
    return results


def make_prediction(model, scaler, input_data, label_encoder=None):
    """
    Make predictions on new data.
    
    Parameters
    ----------
    model : trained sklearn model
    scaler : trained scaler
    input_data : pd.DataFrame with same columns as training data
    label_encoder : trained label encoder (optional, to return string labels)
    """
    # Scale input data
    X_scaled = scaler.transform(input_data)
    
    # Predict
    preds = model.predict(X_scaled)
    
    # Get confidence
    try:
        probs = model.predict_proba(X_scaled)
        confidences = np.max(probs, axis=1)
    except AttributeError:
        confidences = np.ones(len(preds))
        
    # Map back to labels if encoder provided
    if label_encoder:
        pred_labels = label_encoder.inverse_transform(preds)
        return pred_labels, confidences
    
    return preds, confidences
