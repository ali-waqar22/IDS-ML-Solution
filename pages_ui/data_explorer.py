import streamlit as st
import sys
import os
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessor import get_dataset_stats

def render():
    st.markdown('<div class="main-title">DATA EXPLORER</div>', unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("No dataset loaded. Please go to the Dashboard to generate or upload data.")
        return
        
    df = st.session_state.dataset
    
    st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
    st.write(f"### Dataset Preview: {st.session_state.dataset_name}")
    st.dataframe(df.head(100), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
        st.write("### Dataset Statistics")
        stats = get_dataset_stats(df)
        
        st.write(f"**Total Rows:** {stats['rows']:,}")
        st.write(f"**Total Columns:** {stats['columns']}")
        st.write(f"**Numeric Features:** {stats['numeric_cols']}")
        st.write(f"**Categorical Features:** {stats['categorical_cols']}")
        st.write(f"**Missing Values:** {stats['missing_values']}")
        
        if stats['missing_values'] > 0:
            st.warning(f"Found {stats['missing_values']} missing values. The preprocessor will handle these during training.")
        else:
            st.success("No missing values found. Data is clean.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='css-1r6slb0'>", unsafe_allow_html=True)
        st.write("### Class Distribution (Label)")
        if 'Label' in df.columns:
            dist = df['Label'].value_counts().reset_index()
            dist.columns = ['Attack Type', 'Count']
            
            # Show as bar chart
            st.bar_chart(dist.set_index('Attack Type'))
            
            # Show as table
            dist['Percentage'] = (dist['Count'] / len(df) * 100).round(2).astype(str) + '%'
            st.dataframe(dist, use_container_width=True)
        else:
            st.error("No 'Label' column found in dataset. A target column named 'Label' is required for classification.")
        st.markdown("</div>", unsafe_allow_html=True)
