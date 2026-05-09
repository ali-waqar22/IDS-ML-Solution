import streamlit as st
import sys
import os
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessor import preprocess_dataset
from utils.model_manager import get_model, train_and_evaluate

def render():
    st.markdown('<div class="main-title">MODEL TRAINING</div>', unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("No dataset loaded. Please go to the Dashboard to generate or upload data.")
        return
        
    if 'Label' not in st.session_state.dataset.columns:
        st.error("Dataset must contain a 'Label' column for training.")
        return
        
    df = st.session_state.dataset
    
    # ── 1. Model Selection ──
    st.markdown("### 1. Select Algorithm")
    cols = st.columns(4)
    
    models = ['k-NN', 'Decision Tree', 'Random Forest', 'Naive Bayes']
    
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'Random Forest'
        
    for i, model_name in enumerate(models):
        with cols[i]:
            # Use buttons to select model, update session state
            is_selected = st.session_state.selected_model == model_name
            btn_type = "primary" if is_selected else "secondary"
            
            if st.button(model_name, key=f"btn_select_{model_name}", use_container_width=True, type=btn_type):
                st.session_state.selected_model = model_name
                st.rerun()
                
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── 2. Algorithm Configuration ──
    st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
    st.write("### Algorithm Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    params = {}
    selected = st.session_state.selected_model
    
    with config_col1:
        st.write("**k-NN Settings**" if selected == 'k-NN' else "k-NN (Disabled)")
        k = st.slider("k (Neighbors)", 1, 20, 5, disabled=(selected != 'k-NN'))
        metric = st.radio("Metric", ['Euclidean', 'Manhattan', 'Cosine'], disabled=(selected != 'k-NN'))
        if selected == 'k-NN':
            params = {'k': k, 'metric': metric}
            
    with config_col2:
        st.write("**Tree Settings**" if selected in ['Decision Tree', 'Random Forest'] else "DT/RF (Disabled)")
        max_depth = st.slider("Max Depth", 1, 50, 10, disabled=(selected not in ['Decision Tree', 'Random Forest']))
        criterion = st.radio("Criterion", ['Gini', 'Entropy'], disabled=(selected not in ['Decision Tree', 'Random Forest']))
        
        n_trees = None
        if selected == 'Random Forest':
            n_trees = st.slider("n_trees", 10, 200, 100)
            params = {'max_depth': max_depth, 'criterion': criterion, 'n_estimators': n_trees}
        elif selected == 'Decision Tree':
            params = {'max_depth': max_depth, 'criterion': criterion}
            
    with config_col3:
        st.write("**Naive Bayes Settings**" if selected == 'Naive Bayes' else "NB (Disabled)")
        nb_type = st.radio("Type", ['Gaussian', 'Multinomial'], disabled=(selected != 'Naive Bayes'))
        var_smooth = st.text_input("Var Smoothing", "1e-9", disabled=(selected != 'Naive Bayes'))
        if selected == 'Naive Bayes':
            params = {'var_smoothing': var_smooth, 'type': nb_type}
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ── 3. Training Settings & Execution ──
    st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
    st.write("### Training Settings")
    
    split_col, cv_col = st.columns(2)
    with split_col:
        train_split = st.slider("Train/Test Split (%)", 50, 90, 70, step=5)
    with cv_col:
        scale_method = st.selectbox("Scaling Method", ["Standard Scaler", "MinMax Scaler"])
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 TRAIN MODEL", use_container_width=True, type="primary"):
        with st.spinner(f"Preprocessing data and training {st.session_state.selected_model}..."):
            try:
                # Preprocessing
                test_size = 1.0 - (train_split / 100.0)
                sm = 'standard' if scale_method == "Standard Scaler" else 'minmax'
                
                prep_data = preprocess_dataset(df, target_col='Label', test_size=test_size, scale_method=sm)
                st.session_state.preprocessed_data = prep_data
                
                # Training
                model_obj = get_model(st.session_state.selected_model, params)
                
                results = train_and_evaluate(
                    model_obj, 
                    prep_data['X_train'], prep_data['y_train'],
                    prep_data['X_test'], prep_data['y_test'],
                    class_names=prep_data['class_names']
                )
                
                # Save model to session
                model_id = f"{st.session_state.selected_model}_{pd.Timestamp.now().strftime('%H%M%S')}"
                
                st.session_state.trained_models[model_id] = {
                    'name': st.session_state.selected_model,
                    'params': params,
                    'results': results,
                    'preprocessor': prep_data, # Contains scaler and encoders
                    'dataset': st.session_state.dataset_name,
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                st.session_state.active_model_name = model_id
                
                st.success(f"Model trained successfully! Accuracy: {results['accuracy']*100:.2f}%")
                
                # Show mini-results
                st.write("#### Quick Results Summary")
                res_cols = st.columns(4)
                res_cols[0].metric("Accuracy", f"{results['accuracy']*100:.2f}%")
                res_cols[1].metric("Precision (Macro)", f"{results['precision']*100:.2f}%")
                res_cols[2].metric("Recall (Macro)", f"{results['recall']*100:.2f}%")
                res_cols[3].metric("F1 Score", f"{results['f1']*100:.2f}%")
                
                st.info("📊 View detailed charts in the **Visualizations** tab.")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                
    st.markdown("</div>", unsafe_allow_html=True)
