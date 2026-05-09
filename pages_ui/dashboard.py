import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_generator import generate_ids_dataset

def render():
    st.markdown('<div class="main-title">DASHBOARD OVERVIEW</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Network Intrusion Detection System (NIDS) Machine Learning Dashboard**.
    
    This platform allows you to:
    1. Load or generate network traffic datasets.
    2. Preprocess the data for machine learning.
    3. Train and compare various classification models.
    4. Visualize performance and make predictions on new network traffic.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Quick Start Options")
        
        # Option 1: Generate Synthetic Data
        st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
        st.write("#### 1. Generate Synthetic CICIDS2017 Dataset")
        st.write("Generate a realistic dataset modeling web attacks, DDoS, and infiltrations.")
        
        n_samples = st.slider("Number of samples to generate:", 1000, 20000, 5000, 1000)
        
        if st.button("Generate Dataset", key="btn_generate"):
            with st.spinner("Generating complex network flow data..."):
                df = generate_ids_dataset(n_samples=n_samples)
                st.session_state.dataset = df
                st.session_state.dataset_name = f"CICIDS2017_synthetic_{n_samples}.csv"
                
                # Clear preprocessed data if dataset changes
                st.session_state.preprocessed_data = None
                
            st.success(f"Generated dataset with {len(df)} records!")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("### &nbsp;") # Spacing alignment
        
        # Option 2: Upload Data
        st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
        st.write("#### 2. Upload Custom Dataset")
        st.write("Upload your own CSV network logs for analysis.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.dataset = df
                st.session_state.dataset_name = uploaded_file.name
                
                # Clear preprocessed data
                st.session_state.preprocessed_data = None
                
                st.success(f"Successfully loaded {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Dataset Status
    if st.session_state.dataset is not None:
        st.markdown("### Current Workspace Status")
        df = st.session_state.dataset
        
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Dataset", st.session_state.dataset_name)
        mcol2.metric("Total Records", f"{len(df):,}")
        mcol3.metric("Features", len(df.columns) - 1 if 'Label' in df.columns else len(df.columns))
        
        if 'Label' in df.columns:
            mcol4.metric("Attack Classes", df['Label'].nunique())
        
        st.info("💡 Next Step: Head to the **Model Training** tab to build your NIDS classifiers.")
