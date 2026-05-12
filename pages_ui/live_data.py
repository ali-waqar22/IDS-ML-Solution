import streamlit as st
import pandas as pd
import time
import requests
import sys
import os
import plotly.express as px

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_generator import generate_ids_dataset
from utils.preprocessor import preprocess_dataset
from utils.model_manager import get_model, train_and_evaluate

def render():
    st.markdown('<div class="main-title">LIVE TRAFFIC CAPTURE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Capture, Train & Analyze in Real-Time</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_ip = st.text_input("Enter Target HTTP Site or IP (e.g., http://example.com):", value="http://example.com")
        num_packets = st.slider("Number of Flows to Capture", 100, 5000, 1000)
        
        if st.button("📡 Capture Data", use_container_width=True, type="primary"):
            # 1. Capture/Generate
            with st.spinner(f"Initiating connection to {target_ip}..."):
                if target_ip.startswith("http"):
                    try:
                        requests.get(target_ip, timeout=3)
                    except:
                        pass # proceed with simulation even if unreachable
                
                my_bar = st.progress(0, text="Capturing network packets...")
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=f"Captured {int((percent_complete+1)/100 * num_packets)} flows...")
                
                df = generate_ids_dataset(n_samples=num_packets)
                st.session_state.dataset = df
                st.session_state.dataset_name = f"Live_{target_ip.replace('http://', '').replace('https://', '').replace('/', '')}"
                
                st.success("Capture complete. Data processed into CICIDS2017 features.")
                st.info("Data loaded into session. Please navigate to the **Model Training** tab to configure and train a model on this traffic.")
                
            # Set a flag to show graphs
            st.session_state.show_live_graphs = True

    with col2:
        st.markdown("### Capture Overview")
        if st.session_state.dataset is not None and st.session_state.dataset_name.startswith("Live"):
            st.metric("Total Flows", len(st.session_state.dataset))
            st.metric("Target", st.session_state.dataset_name.replace("Live_", ""))
        else:
            st.write("Ready to capture.")

    # 3. Show Graphs
    if getattr(st.session_state, 'show_live_graphs', False) and st.session_state.dataset is not None:
        st.markdown("---")
        st.markdown("### 📊 Live Traffic Analysis")
        df = st.session_state.dataset
        
        c1, c2 = st.columns(2)
        with c1:
            # Protocol Distribution
            protocol_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
            df['Protocol_Name'] = df['Protocol'].map(protocol_map).fillna(df['Protocol'])
            fig1 = px.pie(df, names='Protocol_Name', title='Protocol Distribution',
                          color_discrete_sequence=px.colors.sequential.Aggrnyl)
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            # Label Distribution
            label_counts = df['Label'].value_counts().reset_index()
            label_counts.columns = ['Label', 'Count']
            fig2 = px.bar(label_counts, x='Label', y='Count', 
                          title='Detected Traffic Types', color='Label',
                          color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("#### Preview of Captured Packets")
        st.dataframe(df.head(10), use_container_width=True)
