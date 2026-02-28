#!/usr/bin/env python3
"""
Streamlit Dashboard for Telecom Customer Segmentation.
Provides interactive customer segment prediction and visualization.

Student: Kongnyuy Raymond Afoni (FE25P028)
Course: Artificial Intelligence and Machine Learning Fundamentals
Institution: University of Buea
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import base64
from PIL import Image

# Import configuration
from src.config import (
    MODEL_PATH, PREPROCESSOR_PATH, FIGURES_DIR,
    CLUSTER_PROFILES_PATH, BUSINESS_ACTIONS_PATH,
    STUDENT_NAME, STUDENT_ID, PROGRAM, INSTITUTION,
    FACULTY, DEPARTMENT, COURSE, INSTRUCTOR, DATE,
    PROJECT_TITLE, HIGH_CHURN_THRESHOLD, MEDIUM_CHURN_THRESHOLD
)

# Page configuration
st.set_page_config(
    page_title="FE25P028 EEN637 Telecom Customer Segmentation",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved readability and professional design
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        background-color: #1F497D;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Cluster result box */
    .cluster-result {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 10px;
        border-left: 6px solid #1F497D;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        color: #212529;
    }
    .cluster-result h3 {
        color: #1F497D;
        margin-top: 0;
        font-weight: 600;
    }
    .cluster-result p {
        margin: 8px 0;
        font-size: 1.1rem;
    }
    
    /* Risk-specific variations */
    .high-risk {
        border-left-color: #c0392b;
        background-color: #fdedec;
    }
    .high-risk h3 {
        color: #c0392b;
    }
    .medium-risk {
        border-left-color: #f39c12;
        background-color: #fef5e7;
    }
    .medium-risk h3 {
        color: #e67e22;
    }
    .low-risk {
        border-left-color: #27ae60;
        background-color: #e8f8f5;
    }
    .low-risk h3 {
        color: #27ae60;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
        color: #2c3e50;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .metric-card h4 {
        color: #1F497D;
        margin-bottom: 15px;
        font-weight: 600;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 8px;
    }
    .metric-card p {
        margin: 8px 0;
        font-size: 1rem;
        color: #34495e;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Improve general text readability */
    body {
        color: #2c3e50;
    }
    h1, h2, h3, h4 {
        color: #1F497D;
    }
    .stButton button {
        background-color: #1F497D;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #153a5e;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_preprocessor():
    """Load trained model and preprocessor."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data
def load_cluster_profiles():
    """Load cluster profiles."""
    try:
        return pd.read_csv(CLUSTER_PROFILES_PATH)
    except:
        return None


@st.cache_data
def load_business_actions():
    """Load business actions."""
    try:
        return pd.read_csv(BUSINESS_ACTIONS_PATH)
    except:
        return None


def get_churn_risk_level(churn_rate):
    """Determine churn risk level."""
    if churn_rate >= HIGH_CHURN_THRESHOLD:
        return "High", "high-risk"
    elif churn_rate >= MEDIUM_CHURN_THRESHOLD:
        return "Medium", "medium-risk"
    else:
        return "Low", "low-risk"


def create_input_form():
    """Create sidebar input form for customer data."""
    st.sidebar.header("📋 Customer Information")
    
    with st.sidebar.form("customer_input"):
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", 
                               ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check",
                                      "Bank transfer (automatic)", 
                                      "Credit card (automatic)"])
        
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", 
                                     ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", 
                                       ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", 
                                      ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", 
                                    ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", 
                                        ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", 
                                   ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", 
                                   ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", 
                                       ["No", "Yes", "No internet service"])
        
        st.subheader("Charges")
        monthly_charges = st.number_input("Monthly Charges ($)", 
                                         min_value=0.0, max_value=150.0, 
                                         value=50.0, step=0.1)
        
        submitted = st.form_submit_button("🔍 Predict Segment", 
                                         width='stretch')
    
    if submitted:
        return {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment_method,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': monthly_charges * (tenure + 1),  # Approximation
            'customerID': 'NEW-CUSTOMER-001',
            'Churn': 'No'  # Placeholder
        }
    return None


def prediction_tab():
    """Prediction tab content."""
    st.markdown('<div class="main-header">📡 Customer Segment Predictor</div>', 
                unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{PROJECT_TITLE}</div>', 
                unsafe_allow_html=True)
    
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor()
    cluster_profiles = load_cluster_profiles()
    business_actions = load_business_actions()
    
    if model is None or preprocessor is None:
        st.warning("⚠️ Model not found. Please run 'python main.py' first to train the model.")
        return
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("👈 Enter customer details in the sidebar to predict their segment.")
        
        # Get input
        input_data = create_input_form()
    
    with col2:
        if input_data:
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Transform using preprocessor
            try:
                processed = preprocessor.transform(input_df)
                feature_cols = preprocessor.get_feature_names()
                X = processed[feature_cols].values
                
                # Predict
                cluster_id = model.predict(X)[0]
                
                # Get cluster info
                if cluster_profiles is not None:
                    cluster_info = cluster_profiles[cluster_profiles['cluster_id'] == cluster_id]
                    if not cluster_info.empty:
                        cluster_name = cluster_info.iloc[0].get('cluster_name', f'Cluster {cluster_id}')
                        churn_rate = cluster_info.iloc[0].get('churn_rate', 0)
                        size_pct = cluster_info.iloc[0].get('size_pct', 0)
                    else:
                        cluster_name = f'Cluster {cluster_id}'
                        churn_rate = 0
                        size_pct = 0
                else:
                    cluster_name = f'Cluster {cluster_id}'
                    churn_rate = 0
                    size_pct = 0
                
                # Get business action
                if business_actions is not None:
                    action_info = business_actions[business_actions['cluster_id'] == cluster_id]
                    if not action_info.empty:
                        marketing = action_info.iloc[0].get('marketing_action', '')
                        retention = action_info.iloc[0].get('retention_strategy', '')
                    else:
                        marketing = retention = ''
                else:
                    marketing = retention = ''
                
                # Determine risk level
                risk_level, risk_class = get_churn_risk_level(churn_rate)
                
                # Display result
                st.markdown(f"""
                <div class="cluster-result {risk_class}">
                    <h3>🎯 Predicted Segment: {cluster_name}</h3>
                    <p><strong>Cluster ID:</strong> {cluster_id}</p>
                    <p><strong>Segment Size:</strong> {size_pct:.1f}% of customer base</p>
                    <p><strong>Churn Rate:</strong> {churn_rate*100:.1f}%</p>
                    <p><strong>Risk Level:</strong> <span style="font-weight:bold;color:{'#c0392b' if risk_level=='High' else '#e67e22' if risk_level=='Medium' else '#27ae60'}">{risk_level}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.subheader("📋 Recommended Actions")
                
                col_mkt, col_ret = st.columns(2)
                with col_mkt:
                    st.markdown("**Marketing Action**")
                    st.info(marketing if marketing else "No specific marketing action defined.")
                
                with col_ret:
                    st.markdown("**Retention Strategy**")
                    st.info(retention if retention else "No specific retention strategy defined.")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.info("👈 Fill in the customer details and click 'Predict Segment' to see results.")


def segment_overview_tab():
    """Segment overview tab content."""
    st.header("📊 Segment Overview")
    
    cluster_profiles = load_cluster_profiles()
    
    if cluster_profiles is not None:
        # Display profiles table
        st.subheader("Cluster Profiles")
        
        # Select columns to display
        display_cols = ['cluster_id', 'cluster_name', 'size', 'size_pct', 'churn_rate']
        available_cols = [c for c in display_cols if c in cluster_profiles.columns]
        
        st.dataframe(cluster_profiles[available_cols], width='stretch')
        
        # Key metrics
        st.subheader("Key Metrics by Segment")
        
        cols = st.columns(len(cluster_profiles))
        for i, (_, row) in enumerate(cluster_profiles.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{row.get('cluster_name', f'Cluster {row["cluster_id"]}')}</h4>
                    <p><strong>Size:</strong> {row.get('size', 0)} customers</p>
                    <p><strong>Churn:</strong> {row.get('churn_rate', 0)*100:.1f}%</p>
                    <p><strong>Tenure:</strong> {row.get('tenure_mean', 0):.1f} mo</p>
                    <p><strong>Monthly:</strong> ${row.get('MonthlyCharges_mean', 0):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Cluster profiles not found. Please run the training pipeline first.")


def feature_importance_tab():
    """Feature importance tab content."""
    st.header("🔍 Feature Importance")
    
    # Try to load ANOVA results
    try:
        anova_path = Path("reports/anova_results.csv")
        if anova_path.exists():
            anova_results = pd.read_csv(anova_path)
            st.subheader("Top Discriminating Features (ANOVA F-Statistic)")
            st.dataframe(anova_results.head(15), width='stretch')
        else:
            st.info("ANOVA results will be available after running the full pipeline.")
    except:
        st.info("Feature importance data not available.")
    
    # Display feature importance plot
    importance_plot = FIGURES_DIR / "24_feature_importance_bar.png"
    if importance_plot.exists():
        st.subheader("Feature Importance Visualization")
        st.image(str(importance_plot), width='stretch')


def all_plots_tab():
    """Gallery tab showing all generated plots."""
    st.header("📈 All Visualizations")
    
    # Get all PNG files
    if FIGURES_DIR.exists():
        png_files = sorted(FIGURES_DIR.glob("*.png"))
        
        if png_files:
            st.write(f"Found {len(png_files)} visualizations:")
            
            # Display in 2-column grid
            cols = st.columns(2)
            for i, png_file in enumerate(png_files):
                with cols[i % 2]:
                    st.subheader(png_file.stem.replace('_', ' ').title())
                    st.image(str(png_file), width='stretch')
        else:
            st.info("No visualizations found. Please run the training pipeline first.")
    else:
        st.info("Figures directory not found.")


def about_tab():
    """About tab content."""
    st.header("ℹ️ About This Project")
    
    st.markdown(f"""
    ### {PROJECT_TITLE}
    
    **Student:** {STUDENT_NAME} ({STUDENT_ID})  
    **Program:** {PROGRAM}  
    **Institution:** {INSTITUTION}  
    **Faculty:** {FACULTY}  
    **Department:** {DEPARTMENT}  
    **Course:** {COURSE}  
    **Instructor:** {INSTRUCTOR}  
    **Date:** {DATE}
    
    ### Project Overview
    
    This project applies machine learning clustering algorithms (K-Means, Agglomerative, DBSCAN) 
    to segment telecom customers based on their usage patterns, demographics, and service subscriptions. 
    The goal is to enable smarter marketing strategies and improve customer retention.
    
    ### Methodology
    
    1. **Data Loading & Splitting:** Stratified train/validation/test split (60/20/20)
    2. **Exploratory Data Analysis:** Three composite figures revealing key patterns
    3. **Preprocessing:** Feature engineering, encoding, and scaling (fit on training only)
    4. **Model Selection:** Evaluation of K-Means, Agglomerative, and DBSCAN
    5. **Final Training:** Retraining on combined train+validation
    6. **Evaluation:** Internal metrics + external validation using churn
    7. **Profiling:** Cluster characterization and empirical naming
    8. **Business Actions:** Marketing and retention strategies per segment
    
    ### Technologies Used
    
    - Python 3.9+
    - scikit-learn (clustering algorithms)
    - pandas & numpy (data manipulation)
    - matplotlib & seaborn (visualization)
    - Streamlit (dashboard)
    
    ### Dashboard Features
    
    - **Prediction:** Input customer details to predict their segment
    - **Segment Overview:** View cluster profiles and key metrics
    - **Feature Importance:** Understand which features drive segmentation
    - **All Plots:** Browse all generated visualizations
    

    """)


def main():
    """Main dashboard function."""
    # Create tabs
    tabs = st.tabs([
        "🔮 Prediction", 
        "📊 Segment Overview", 
        "🔍 Feature Importance",
        "📈 All Plots",
        "ℹ️ About"
    ])
    
    with tabs[0]:
        prediction_tab()
    
    with tabs[1]:
        segment_overview_tab()
    
    with tabs[2]:
        feature_importance_tab()
    
    with tabs[3]:
        all_plots_tab()
    
    with tabs[4]:
        about_tab()
    
    # Footer
    st.markdown(f"""
    <div class="footer">
        <p>{PROJECT_TITLE}</p>
        <p>{STUDENT_NAME} ({STUDENT_ID}) | {INSTITUTION} | {DATE}</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()