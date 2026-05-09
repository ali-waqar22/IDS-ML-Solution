"""
plotters.py — Visualization Functions
=====================================
Contains Plotly and Seaborn visualization functions for the dashboard.
Designed to match the blue/white color scheme of the UI.
"""

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

# Standard theme colors for consistency
COLORS = {
    'primary': '#2563EB',      # Royal Blue
    'secondary': '#3B82F6',    # Lighter Blue
    'accent': '#60A5FA',       # Very Light Blue
    'text': '#1E293B',         # Slate Dark
    'background': '#FFFFFF',
    'grid': '#F1F5F9',
    'success': '#10B981',      # Emerald Green
    'warning': '#F59E0B',      # Amber
    'danger': '#EF4444'        # Red
}

def plot_class_distribution(df, target_col):
    """Plot the distribution of classes in the dataset."""
    counts = df[target_col].value_counts().reset_index()
    counts.columns = ['Class', 'Count']
    
    fig = px.bar(
        counts, x='Class', y='Count',
        color='Class',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Class Distribution"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=None,
    )
    return fig

def plot_correlation_matrix(df, max_features=15):
    """Plot a correlation heatmap of the numerical features."""
    # Select numerical columns
    num_df = df.select_dtypes(include=[np.number])
    
    # If too many features, select a subset with highest variance
    if len(num_df.columns) > max_features:
        vars_ = num_df.var().sort_values(ascending=False)
        selected_cols = vars_.head(max_features).index
        num_df = num_df[selected_cols]
        
    corr = num_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Blues',
        zmin=-1, zmax=1
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix (Top Variance Features)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

def plot_confusion_matrix(cm, class_names):
    """Plot a stylized confusion matrix."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title="Confusion Matrix",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig

def plot_feature_importance(importances, feature_names, top_n=10):
    """Plot feature importance for tree-based models."""
    if importances is None:
        return None
        
    # Sort importances
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Reverse to have highest at top of bar chart
    top_indices = indices[::-1]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    fig = go.Figure(go.Bar(
        x=top_importances,
        y=top_features,
        orientation='h',
        marker_color=COLORS['primary']
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Importance Score",
        yaxis_title=None
    )
    return fig

def plot_roc_curve(y_test, y_prob, class_names):
    """Plot Multi-class ROC Curve."""
    if y_prob is None:
        return None
        
    # Binarize the output for multi-class
    from sklearn.preprocessing import label_binarize
    classes = list(range(len(class_names)))
    y_test_bin = label_binarize(y_test, classes=classes)
    
    n_classes = len(classes)
    
    fig = go.Figure()
    
    # Plot ROC curve for each class
    colors = px.colors.qualitative.Plotly
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{class_names[i]} (AUC = {roc_auc:.2f})",
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
        
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random Guess",
        mode='lines',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='False Positive Rate', gridcolor=COLORS['grid']),
        yaxis=dict(title='True Positive Rate', gridcolor=COLORS['grid']),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    
    return fig

def plot_decision_boundary_2d(model, X, y, feature_x, feature_y, scaler, feature_names):
    """
    Visualize decision boundary using only 2 features.
    Warning: This requires retraining a model on just 2 features.
    """
    # Create meshgrid
    h = .02
    x_min, x_max = X[feature_x].min() - 1, X[feature_x].max() + 1
    y_min, y_max = X[feature_y].min() - 1, X[feature_y].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # This is a complex visualization to do correctly dynamically in Streamlit.
    # Often it's better to just show scatter plot of classes.
    
    fig = px.scatter(
        X, x=feature_x, y=feature_y, color=y.astype(str),
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title=f"Scatter Plot: {feature_x} vs {feature_y}"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig
