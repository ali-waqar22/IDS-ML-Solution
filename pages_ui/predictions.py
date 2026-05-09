import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_manager import make_prediction

def render():
    st.markdown('<div class="main-title">MAKE PREDICTIONS</div>', unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("No models trained yet. Please go to the Model Training tab first.")
        return
        
    # Model Selection
    model_ids = list(st.session_state.trained_models.keys())
    
    if st.session_state.active_model_name not in model_ids:
        st.session_state.active_model_name = model_ids[-1]
        
    active_id = st.session_state.active_model_name
    active_model_data = st.session_state.trained_models[active_id]
    model = active_model_data['results']['model']
    prep_data = active_model_data['preprocessor']
    
    st.write(f"**Active Model:** {active_model_data['name']} (ID: {active_id})")
    
    # ── Manual Input ──
    st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
    st.write("### Manual Input")
    
    features = prep_data['feature_names']
    
    # Show only first 6 features for manual input to avoid overwhelming UI
    display_features = features[:6]
    
    # Create input fields
    inputs = {}
    cols = st.columns(3)
    
    for i, feature in enumerate(display_features):
        with cols[i % 3]:
            # Get mean value from raw training data as default
            default_val = float(prep_data['X_train_raw'][feature].mean())
            inputs[feature] = st.number_input(f"{feature}", value=default_val, format="%.2f")
            
    # For remaining features, use their mean silently
    for feature in features[6:]:
        inputs[feature] = float(prep_data['X_train_raw'][feature].mean())
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("PREDICT", type="primary"):
        input_df = pd.DataFrame([inputs])
        
        # Predict using all trained models for comparison
        predictions = {}
        confidences = {}
        
        for mid, mdata in st.session_state.trained_models.items():
            m = mdata['results']['model']
            p_data = mdata['preprocessor']
            
            pred, conf = make_prediction(
                m, p_data['scaler'], input_df, p_data['label_encoder']
            )
            predictions[mdata['name']] = pred[0]
            confidences[mdata['name']] = conf[0] * 100
            
        # Main Prediction Result
        main_pred = predictions[active_model_data['name']]
        main_conf = confidences[active_model_data['name']]
        
        st.success(f"**Result:** {main_pred} ({main_conf:.0f}% confidence)")
        
        # Comparison string
        comp_str = " | ".join([f"{name}: {predictions[name]} ({confidences[name]:.0f}%)" 
                              for name in predictions if name != active_model_data['name']])
        if comp_str:
            st.caption(f"Other models: {comp_str}")
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ── Batch Prediction ──
    st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
    st.write("### Batch Prediction (Upload CSV)")
    
    uploaded_file = st.file_uploader("Browse...", type="csv", key="batch_upload")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows for prediction.")
            
            # Check if all features exist
            missing_cols = [col for col in features if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in uploaded data: {', '.join(missing_cols[:5])}...")
            else:
                if st.button("RUN BATCH PREDICTION"):
                    # Extract just the features needed
                    X_batch = batch_df[features]
                    
                    # Predict
                    pred, conf = make_prediction(
                        model, prep_data['scaler'], X_batch, prep_data['label_encoder']
                    )
                    
                    # Create results dataframe
                    results_df = batch_df.copy()
                    results_df['Final_Pred'] = pred
                    results_df['Confidence'] = [f"{c*100:.0f}%" for c in conf]
                    
                    st.dataframe(results_df.head(), use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "DOWNLOAD RESULTS",
                        csv,
                        "predictions_output.csv",
                        "text/csv",
                        type="primary"
                    )
        except Exception as e:
            st.error(f"Error processing batch file: {str(e)}")
            
    st.markdown("</div>", unsafe_allow_html=True)
