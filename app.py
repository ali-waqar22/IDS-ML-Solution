import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Machine Learning Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match the UI screenshots
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #F0F4F8;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Title */
    .main-title {
        color: #1E293B;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 2px;
        font-family: 'Inter', sans-serif;
    }
    .sub-title {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 30px;
    }
    
    /* Cards/Containers */
    div.css-1r6slb0, div.css-12oz5g7 {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #E2E8F0;
    }
    
    /* Sidebar Navigation Links */
    .nav-link {
        display: flex;
        align-items: center;
        padding: 10px 15px;
        color: #64748B;
        text-decoration: none;
        border-radius: 6px;
        margin-bottom: 5px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .nav-link:hover {
        background-color: #F8FAFC;
        color: #2563EB;
    }
    .nav-link.active {
        background-color: #EFF6FF;
        color: #2563EB;
        border-left: 3px solid #2563EB;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    
    /* Model Selection Buttons */
    .model-btn {
        border: 1px solid #E2E8F0;
        background-color: white;
        color: #64748B;
        border-radius: 6px;
        padding: 10px;
        text-align: center;
        cursor: pointer;
    }
    .model-btn.selected {
        border-color: #2563EB;
        color: #2563EB;
        font-weight: 600;
        box-shadow: 0 0 0 1px #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'active_model_name' not in st.session_state:
    st.session_state.active_model_name = None

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="main-title">MACHINE LEARNING DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">v2.0 - Streamlit</div>', unsafe_allow_html=True)
    
    st.write("") # Spacer
    
    # Navigation items using buttons styled to look like links
    pages = {
        'Dashboard': '🏠',
        'Data Explorer': '📊',
        'Model Training': '⚙️',
        'Visualizations': '📈',
        'Predictions': '🎯',
        'Model History': '📋',
        'Settings': '🔧'
    }
    
    for page_name, icon in pages.items():
        # Handle state changes
        if st.sidebar.button(f"{icon} {page_name}", use_container_width=True, 
                             type="primary" if st.session_state.page == page_name else "secondary"):
            st.session_state.page = page_name
            st.rerun()

    st.write("") # Spacer
    st.write("") # Spacer
    
    # Bottom info section
    if st.session_state.dataset is not None:
        st.markdown(f"**Current Dataset:** {st.session_state.dataset_name}")
        rows, cols = st.session_state.dataset.shape
        # Assuming 'Label' is the target class
        classes = st.session_state.dataset['Label'].nunique() if 'Label' in st.session_state.dataset.columns else "Unknown"
        st.caption(f"Rows: {rows} | Features: {cols-1} | Classes: {classes}")
    
    if st.sidebar.button("NEW SESSION", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Router
if st.session_state.page == 'Dashboard':
    import pages_ui.dashboard as ui
    ui.render()
elif st.session_state.page == 'Data Explorer':
    import pages_ui.data_explorer as ui
    ui.render()
elif st.session_state.page == 'Model Training':
    import pages_ui.model_training as ui
    ui.render()
elif st.session_state.page == 'Visualizations':
    import pages_ui.visualizations as ui
    ui.render()
elif st.session_state.page == 'Predictions':
    import pages_ui.predictions as ui
    ui.render()
else:
    st.title(st.session_state.page)
    st.info(f"{st.session_state.page} functionality is under development.")
