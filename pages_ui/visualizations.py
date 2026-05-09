import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plotters import plot_confusion_matrix, plot_feature_importance, plot_roc_curve

def render():
    st.markdown('<div class="main-title">VISUALIZATION DASHBOARD</div>', unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("No models trained yet. Please go to the Model Training tab first.")
        return
        
    # Active Model Selection
    model_ids = list(st.session_state.trained_models.keys())
    
    # Default to last trained model if none active
    if st.session_state.active_model_name not in model_ids:
        st.session_state.active_model_name = model_ids[-1]
        
    active_id = st.selectbox(
        "Select Model to Visualize", 
        model_ids, 
        index=model_ids.index(st.session_state.active_model_name)
    )
    
    active_model_data = st.session_state.trained_models[active_id]
    results = active_model_data['results']
    prep_data = active_model_data['preprocessor']
    
    # ── Top Row: Comparisons and Importance ──
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
        st.write("### Model Accuracy Comparison")
        
        # Collect all accuracies
        acc_data = []
        for mid, mdata in st.session_state.trained_models.items():
            acc_data.append({
                'Model': mdata['name'],
                'ID': mid,
                'Accuracy': mdata['results']['accuracy'] * 100
            })
            
        df_acc = pd.DataFrame(acc_data)
        
        # Plotly horizontal bar
        fig_acc = px.bar(
            df_acc, 
            x='Accuracy', 
            y='Model', 
            orientation='h',
            text=df_acc['Accuracy'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=['#2563EB']
        )
        fig_acc.update_traces(textposition='outside')
        fig_acc.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=200,
            xaxis_range=[0, 105], # Give room for text
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
        st.write("### Feature Importance")
        
        if results.get('feature_importances') is not None:
            fig_imp = plot_feature_importance(
                results['feature_importances'], 
                prep_data['feature_names'],
                top_n=8
            )
            fig_imp.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info(f"Feature importance not available for {active_model_data['name']}. Try training a Tree or Forest model.")
            # Dummy placeholder for aesthetics matching screenshot
            st.markdown("<div style='height: 180px; display:flex; align-items:center; justify-content:center; color:#64748B;'>Not Applicable</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # ── Middle Row: Decision Boundary Approximation ──
    st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
    st.write("### Feature Scatter Visualization")
    
    scatter_col1, scatter_col2 = st.columns([1, 3])
    
    with scatter_col1:
        st.write("Select Features")
        features = prep_data['feature_names']
        
        # Default features (try to pick numeric)
        default_x = features[0]
        default_y = features[1] if len(features) > 1 else features[0]
        
        x_axis = st.selectbox("X-axis", features, index=features.index(default_x))
        y_axis = st.selectbox("Y-axis", features, index=features.index(default_y))
        
    with scatter_col2:
        # Show scatter of actual test data points colored by predicted class
        # This is a proxy for decision boundary which is complex in high dims
        df_test_plot = prep_data['X_test_raw'].copy()
        
        # Map predicted labels back to original strings
        y_pred_labels = prep_data['label_encoder'].inverse_transform(results['y_pred'])
        df_test_plot['Predicted_Class'] = y_pred_labels
        
        fig_scatter = px.scatter(
            df_test_plot, 
            x=x_axis, 
            y=y_axis, 
            color='Predicted_Class',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_scatter.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=250,
            plot_bgcolor='#E6F0FA', # Match the blue/green background in screenshot
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ── Bottom Row: Confusion Matrix & ROC ──
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
        st.write("### Confusion Matrix")
        
        cm = results['confusion_matrix']
        class_names = prep_data['class_names']
        
        fig_cm = plot_confusion_matrix(cm, class_names)
        fig_cm.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
        st.write("### ROC Curve")
        
        if results.get('y_prob') is not None:
            fig_roc = plot_roc_curve(prep_data['y_test'], results['y_prob'], prep_data['class_names'])
            fig_roc.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_roc, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("ROC Curve requires probability predictions. Not available for this model configuration.")
            
        st.markdown("</div>", unsafe_allow_html=True)
