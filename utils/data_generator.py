"""
data_generator.py — Synthetic CICIDS2017-style Network Traffic Dataset Generator
=================================================================================
Generates realistic network traffic data with BENIGN and various attack categories.
Features mimic the CICIDS2017 dataset structure used in real-world IDS research.
"""

import numpy as np
import pandas as pd


def generate_ids_dataset(n_samples=5000, random_state=42):
    """
    Generate a synthetic network intrusion detection dataset modeled after CICIDS2017.
    
    Attack types and their real-world descriptions:
    - BENIGN: Normal network traffic (web browsing, email, file transfers)
    - DDoS: Distributed Denial of Service attacks flooding the network
    - Brute_Force: SSH/FTP credential brute-force attempts
    - Web_Attack: SQL Injection, XSS, and command injection attacks
    - Infiltration: Stealthy unauthorized access and data exfiltration
    
    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with network traffic features and 'Label' column.
    """
    np.random.seed(random_state)
    
    # Define attack distribution (realistic imbalance: mostly benign)
    attack_ratios = {
        'BENIGN': 0.50,
        'DDoS': 0.20,
        'Brute_Force': 0.12,
        'Web_Attack': 0.10,
        'Infiltration': 0.08
    }
    
    records = []
    
    for label, ratio in attack_ratios.items():
        n = int(n_samples * ratio)
        data = _generate_traffic(label, n, random_state)
        records.append(data)
    
    df = pd.concat(records, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def _generate_traffic(label, n, seed):
    """Generate feature vectors for a specific traffic type."""
    rng = np.random.RandomState(seed + hash(label) % 10000)
    
    # Traffic profiles define realistic statistical distributions per attack type
    profiles = {
        'BENIGN': {
            'Dst_Port': rng.choice([80, 443, 8080, 53, 22, 25, 110, 993], n),
            'Protocol': rng.choice([6, 17, 1], n, p=[0.7, 0.25, 0.05]),
            'Flow_Duration': rng.exponential(50000, n) + 1000,
            'Tot_Fwd_Pkts': rng.poisson(12, n) + 1,
            'Tot_Bwd_Pkts': rng.poisson(10, n) + 1,
            'TotLen_Fwd_Pkts': rng.exponential(2000, n),
            'TotLen_Bwd_Pkts': rng.exponential(5000, n),
            'Fwd_Pkt_Len_Max': rng.uniform(100, 1500, n),
            'Fwd_Pkt_Len_Mean': rng.uniform(50, 800, n),
            'Bwd_Pkt_Len_Max': rng.uniform(100, 1500, n),
            'Bwd_Pkt_Len_Mean': rng.uniform(50, 900, n),
            'Flow_Byts_per_s': rng.exponential(50000, n),
            'Flow_Pkts_per_s': rng.exponential(200, n),
            'Flow_IAT_Mean': rng.exponential(30000, n) + 500,
            'Flow_IAT_Std': rng.exponential(20000, n),
            'Fwd_IAT_Mean': rng.exponential(40000, n) + 500,
            'Bwd_IAT_Mean': rng.exponential(45000, n) + 500,
            'Fwd_PSH_Flags': rng.binomial(1, 0.3, n),
            'Fwd_URG_Flags': rng.binomial(1, 0.01, n),
            'Fwd_Pkts_per_s': rng.exponential(100, n),
            'Bwd_Pkts_per_s': rng.exponential(80, n),
            'Pkt_Len_Min': rng.uniform(0, 100, n),
            'Pkt_Len_Max': rng.uniform(200, 1500, n),
            'Pkt_Len_Mean': rng.uniform(50, 800, n),
            'Pkt_Len_Std': rng.uniform(10, 400, n),
            'FIN_Flag_Cnt': rng.binomial(1, 0.4, n),
            'SYN_Flag_Cnt': rng.binomial(1, 0.5, n),
            'RST_Flag_Cnt': rng.binomial(1, 0.05, n),
            'ACK_Flag_Cnt': rng.binomial(1, 0.8, n),
            'Down_Up_Ratio': rng.uniform(0.5, 3.0, n),
            'Init_Fwd_Win_Byts': rng.choice([8192, 16384, 29200, 65535], n),
            'Init_Bwd_Win_Byts': rng.choice([8192, 16384, 29200, 65535], n),
        },
        'DDoS': {
            'Dst_Port': rng.choice([80, 443, 53], n, p=[0.5, 0.3, 0.2]),
            'Protocol': rng.choice([6, 17, 1], n, p=[0.4, 0.5, 0.1]),
            'Flow_Duration': rng.exponential(5000, n) + 100,
            'Tot_Fwd_Pkts': rng.poisson(200, n) + 50,
            'Tot_Bwd_Pkts': rng.poisson(5, n),
            'TotLen_Fwd_Pkts': rng.exponential(500, n) * 50,
            'TotLen_Bwd_Pkts': rng.exponential(100, n),
            'Fwd_Pkt_Len_Max': rng.uniform(40, 200, n),
            'Fwd_Pkt_Len_Mean': rng.uniform(40, 100, n),
            'Bwd_Pkt_Len_Max': rng.uniform(0, 100, n),
            'Bwd_Pkt_Len_Mean': rng.uniform(0, 50, n),
            'Flow_Byts_per_s': rng.exponential(500000, n) + 100000,
            'Flow_Pkts_per_s': rng.exponential(5000, n) + 1000,
            'Flow_IAT_Mean': rng.exponential(500, n) + 10,
            'Flow_IAT_Std': rng.exponential(300, n),
            'Fwd_IAT_Mean': rng.exponential(200, n) + 10,
            'Bwd_IAT_Mean': rng.exponential(5000, n) + 100,
            'Fwd_PSH_Flags': rng.binomial(1, 0.1, n),
            'Fwd_URG_Flags': rng.binomial(1, 0.02, n),
            'Fwd_Pkts_per_s': rng.exponential(3000, n) + 500,
            'Bwd_Pkts_per_s': rng.exponential(50, n),
            'Pkt_Len_Min': rng.uniform(40, 60, n),
            'Pkt_Len_Max': rng.uniform(60, 200, n),
            'Pkt_Len_Mean': rng.uniform(40, 100, n),
            'Pkt_Len_Std': rng.uniform(5, 50, n),
            'FIN_Flag_Cnt': rng.binomial(1, 0.1, n),
            'SYN_Flag_Cnt': rng.binomial(1, 0.9, n),
            'RST_Flag_Cnt': rng.binomial(1, 0.3, n),
            'ACK_Flag_Cnt': rng.binomial(1, 0.5, n),
            'Down_Up_Ratio': rng.uniform(0.01, 0.3, n),
            'Init_Fwd_Win_Byts': rng.choice([1024, 2048, 4096], n),
            'Init_Bwd_Win_Byts': rng.choice([0, 1024], n, p=[0.7, 0.3]),
        },
        'Brute_Force': {
            'Dst_Port': rng.choice([22, 21, 3389, 445], n, p=[0.4, 0.3, 0.2, 0.1]),
            'Protocol': np.full(n, 6),
            'Flow_Duration': rng.exponential(3000, n) + 500,
            'Tot_Fwd_Pkts': rng.poisson(8, n) + 3,
            'Tot_Bwd_Pkts': rng.poisson(6, n) + 2,
            'TotLen_Fwd_Pkts': rng.exponential(500, n),
            'TotLen_Bwd_Pkts': rng.exponential(300, n),
            'Fwd_Pkt_Len_Max': rng.uniform(50, 500, n),
            'Fwd_Pkt_Len_Mean': rng.uniform(30, 200, n),
            'Bwd_Pkt_Len_Max': rng.uniform(50, 400, n),
            'Bwd_Pkt_Len_Mean': rng.uniform(30, 150, n),
            'Flow_Byts_per_s': rng.exponential(10000, n),
            'Flow_Pkts_per_s': rng.exponential(500, n) + 100,
            'Flow_IAT_Mean': rng.exponential(2000, n) + 100,
            'Flow_IAT_Std': rng.exponential(1500, n),
            'Fwd_IAT_Mean': rng.exponential(2500, n) + 100,
            'Bwd_IAT_Mean': rng.exponential(3000, n) + 200,
            'Fwd_PSH_Flags': rng.binomial(1, 0.6, n),
            'Fwd_URG_Flags': rng.binomial(1, 0.02, n),
            'Fwd_Pkts_per_s': rng.exponential(300, n) + 50,
            'Bwd_Pkts_per_s': rng.exponential(200, n) + 30,
            'Pkt_Len_Min': rng.uniform(20, 60, n),
            'Pkt_Len_Max': rng.uniform(100, 500, n),
            'Pkt_Len_Mean': rng.uniform(40, 200, n),
            'Pkt_Len_Std': rng.uniform(20, 150, n),
            'FIN_Flag_Cnt': rng.binomial(1, 0.3, n),
            'SYN_Flag_Cnt': rng.binomial(1, 0.7, n),
            'RST_Flag_Cnt': rng.binomial(1, 0.4, n),
            'ACK_Flag_Cnt': rng.binomial(1, 0.7, n),
            'Down_Up_Ratio': rng.uniform(0.3, 1.5, n),
            'Init_Fwd_Win_Byts': rng.choice([4096, 8192, 16384], n),
            'Init_Bwd_Win_Byts': rng.choice([4096, 8192], n),
        },
        'Web_Attack': {
            'Dst_Port': rng.choice([80, 443, 8080, 8443], n, p=[0.4, 0.3, 0.2, 0.1]),
            'Protocol': np.full(n, 6),
            'Flow_Duration': rng.exponential(20000, n) + 2000,
            'Tot_Fwd_Pkts': rng.poisson(15, n) + 5,
            'Tot_Bwd_Pkts': rng.poisson(8, n) + 2,
            'TotLen_Fwd_Pkts': rng.exponential(3000, n) + 500,
            'TotLen_Bwd_Pkts': rng.exponential(8000, n),
            'Fwd_Pkt_Len_Max': rng.uniform(200, 1500, n),
            'Fwd_Pkt_Len_Mean': rng.uniform(100, 800, n),
            'Bwd_Pkt_Len_Max': rng.uniform(200, 1500, n),
            'Bwd_Pkt_Len_Mean': rng.uniform(100, 700, n),
            'Flow_Byts_per_s': rng.exponential(30000, n),
            'Flow_Pkts_per_s': rng.exponential(150, n),
            'Flow_IAT_Mean': rng.exponential(10000, n) + 500,
            'Flow_IAT_Std': rng.exponential(8000, n),
            'Fwd_IAT_Mean': rng.exponential(12000, n) + 500,
            'Bwd_IAT_Mean': rng.exponential(15000, n) + 1000,
            'Fwd_PSH_Flags': rng.binomial(1, 0.7, n),
            'Fwd_URG_Flags': rng.binomial(1, 0.05, n),
            'Fwd_Pkts_per_s': rng.exponential(80, n) + 10,
            'Bwd_Pkts_per_s': rng.exponential(50, n) + 5,
            'Pkt_Len_Min': rng.uniform(20, 80, n),
            'Pkt_Len_Max': rng.uniform(500, 1500, n),
            'Pkt_Len_Mean': rng.uniform(100, 700, n),
            'Pkt_Len_Std': rng.uniform(50, 400, n),
            'FIN_Flag_Cnt': rng.binomial(1, 0.35, n),
            'SYN_Flag_Cnt': rng.binomial(1, 0.5, n),
            'RST_Flag_Cnt': rng.binomial(1, 0.1, n),
            'ACK_Flag_Cnt': rng.binomial(1, 0.85, n),
            'Down_Up_Ratio': rng.uniform(1.0, 5.0, n),
            'Init_Fwd_Win_Byts': rng.choice([8192, 16384, 29200, 65535], n),
            'Init_Bwd_Win_Byts': rng.choice([8192, 16384, 29200], n),
        },
        'Infiltration': {
            'Dst_Port': rng.choice([80, 443, 8080, 4444, 5555, 1234], n),
            'Protocol': rng.choice([6, 17], n, p=[0.8, 0.2]),
            'Flow_Duration': rng.exponential(100000, n) + 10000,
            'Tot_Fwd_Pkts': rng.poisson(20, n) + 3,
            'Tot_Bwd_Pkts': rng.poisson(15, n) + 2,
            'TotLen_Fwd_Pkts': rng.exponential(1500, n),
            'TotLen_Bwd_Pkts': rng.exponential(10000, n) + 1000,
            'Fwd_Pkt_Len_Max': rng.uniform(100, 1000, n),
            'Fwd_Pkt_Len_Mean': rng.uniform(40, 400, n),
            'Bwd_Pkt_Len_Max': rng.uniform(200, 1500, n),
            'Bwd_Pkt_Len_Mean': rng.uniform(100, 800, n),
            'Flow_Byts_per_s': rng.exponential(15000, n),
            'Flow_Pkts_per_s': rng.exponential(50, n),
            'Flow_IAT_Mean': rng.exponential(50000, n) + 5000,
            'Flow_IAT_Std': rng.exponential(40000, n),
            'Fwd_IAT_Mean': rng.exponential(60000, n) + 5000,
            'Bwd_IAT_Mean': rng.exponential(70000, n) + 5000,
            'Fwd_PSH_Flags': rng.binomial(1, 0.4, n),
            'Fwd_URG_Flags': rng.binomial(1, 0.03, n),
            'Fwd_Pkts_per_s': rng.exponential(30, n),
            'Bwd_Pkts_per_s': rng.exponential(20, n),
            'Pkt_Len_Min': rng.uniform(20, 80, n),
            'Pkt_Len_Max': rng.uniform(300, 1500, n),
            'Pkt_Len_Mean': rng.uniform(80, 600, n),
            'Pkt_Len_Std': rng.uniform(30, 350, n),
            'FIN_Flag_Cnt': rng.binomial(1, 0.25, n),
            'SYN_Flag_Cnt': rng.binomial(1, 0.4, n),
            'RST_Flag_Cnt': rng.binomial(1, 0.08, n),
            'ACK_Flag_Cnt': rng.binomial(1, 0.75, n),
            'Down_Up_Ratio': rng.uniform(2.0, 8.0, n),
            'Init_Fwd_Win_Byts': rng.choice([8192, 16384, 65535], n),
            'Init_Bwd_Win_Byts': rng.choice([8192, 16384, 29200], n),
        },
    }
    
    profile = profiles[label]
    df = pd.DataFrame(profile)
    df['Label'] = label
    
    # Add small noise to make data more realistic
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        noise = np.random.normal(0, df[col].std() * 0.02, len(df))
        df[col] = np.abs(df[col] + noise)
    
    return df
