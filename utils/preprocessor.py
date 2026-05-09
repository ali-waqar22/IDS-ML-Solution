"""
preprocessor.py — Data Preprocessing Pipeline for IDS
=====================================================
Handles missing values, encoding, scaling, and train/test splitting
for network intrusion detection data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess_dataset(df, target_col='Label', test_size=0.3, random_state=42, scale_method='standard'):
    """
    Complete preprocessing pipeline for IDS dataset.
    
    Steps:
    1. Handle missing/infinite values
    2. Encode categorical features
    3. Split into train/test sets
    4. Scale numerical features
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with features and target.
    target_col : str
        Name of the target column.
    test_size : float
        Proportion of data for testing (0.0 to 1.0).
    random_state : int
        Random seed for reproducibility.
    scale_method : str
        'standard' for StandardScaler or 'minmax' for MinMaxScaler.
    
    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, scaler,
                    label_encoder, feature_names, class_names, preprocessing_log
    """
    log = []
    df_clean = df.copy()
    
    # ── Step 1: Handle missing and infinite values ──
    initial_rows = len(df_clean)
    
    # Replace infinities with NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    missing_before = df_clean.isnull().sum().sum()
    log.append(f"Missing values found: {missing_before}")
    
    # Fill numeric NaN with median, categorical with mode
    for col in df_clean.columns:
        if col == target_col:
            continue
        if df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Drop rows with missing target
    df_clean.dropna(subset=[target_col], inplace=True)
    rows_dropped = initial_rows - len(df_clean)
    log.append(f"Rows dropped (missing target): {rows_dropped}")
    
    # ── Step 2: Encode categorical features ──
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_clean[target_col])
    class_names = list(label_encoder.classes_)
    log.append(f"Classes found: {class_names}")
    
    # Drop target column from features
    X = df_clean.drop(columns=[target_col])
    
    # Encode remaining categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {}
    for col in cat_cols:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))
        encoders[col] = enc
    log.append(f"Categorical columns encoded: {cat_cols if cat_cols else 'None'}")
    
    feature_names = X.columns.tolist()
    
    # ── Step 3: Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.append(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # ── Step 4: Scale features ──
    if scale_method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=feature_names, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=feature_names, 
        index=X_test.index
    )
    log.append(f"Scaling method: {scale_method}")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'class_names': class_names,
        'preprocessing_log': log,
        'cat_encoders': encoders,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
    }


def get_dataset_stats(df, target_col='Label'):
    """Return key statistics about the dataset for display."""
    stats = {
        'rows': len(df),
        'columns': len(df.columns),
        'features': len(df.columns) - 1,
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns),
    }
    
    if target_col in df.columns:
        stats['classes'] = df[target_col].nunique()
        stats['class_distribution'] = df[target_col].value_counts().to_dict()
    
    return stats
