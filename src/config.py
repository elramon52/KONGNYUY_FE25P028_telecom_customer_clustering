"""
Configuration module for Telecom Customer Segmentation Project.
Contains paths, random seeds, global settings, and student information.
"""

import os
from pathlib import Path

# =============================================================================
# STUDENT INFORMATION
# =============================================================================
STUDENT_NAME = "Kongnyuy Raymond Afoni"
STUDENT_ID = "FE25P028"
PROGRAM = "Masters of Engineering in Telecommunications and Networks"
INSTITUTION = "University of Buea"
FACULTY = "Faculty of Engineering and Technology"
DEPARTMENT = "Department of Electrical and Electronics Engineering"
COURSE = "Artificial Intelligence and Machine Learning Fundamentals"
INSTRUCTOR = "Kinge Mbeke Theophane Osee"
DATE = "February 2026"
PROJECT_TITLE = "Clustering Mobile Customers by Usage Patterns for Smarter Marketing and Retention: A Machine Learning Approach to Customer Segmentation in the Telecom Industry"

# =============================================================================
# RANDOM SEEDS FOR REPRODUCIBILITY
# =============================================================================
RANDOM_STATE = 42
NP_RANDOM_SEED = 42

# =============================================================================
# PATHS – Dynamic based on this file's location
# =============================================================================
# Get the directory of this file (src/)
SRC_DIR = Path(__file__).parent
# Project root is parent of src/
BASE_DIR = SRC_DIR.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = BASE_DIR / "docs"
PRESENTATION_DIR = BASE_DIR / "presentation"

# Data files
DATASET_PATH = DATA_DIR / "telecom_customer_dataset.csv"

# Model files
MODEL_PATH = MODELS_DIR / "final_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# Report files
CLUSTER_PROFILES_PATH = REPORTS_DIR / "cluster_profiles.csv"
BUSINESS_ACTIONS_PATH = REPORTS_DIR / "business_actions.csv"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
TARGET_COL = "Churn"
ID_COL = "customerID"

# Split ratios
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

# =============================================================================
# COLOR PALETTE FOR VISUALIZATIONS
# =============================================================================
COLOR_PALETTE = ['#E63946', '#0e0020', '#a80202', '#048f16', '#3b2653', '#E9C46A']

# =============================================================================
# PLOT CONFIGURATION
# =============================================================================
DPI = 300
FIGURE_FORMAT = "png"

# =============================================================================
# MODEL SELECTION CONFIGURATION
# =============================================================================
KMEANS_K_RANGE = range(2, 11)
AGGLOMERATIVE_K_RANGE = range(2, 11)
AGGLOMERATIVE_LINKAGES = ['ward', 'complete', 'average']
DBSCAN_EPS_VALUES = [0.3, 0.5, 0.7, 1.0]
DBSCAN_MIN_SAMPLES = [3, 5, 7, 10]

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
TENURE_GROUP_BINS = [0, 12, 48, 72]
TENURE_GROUP_LABELS = ['New', 'Mid', 'Loyal']
CONTRACT_MAPPING = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
AUTO_PAYMENT_METHODS = ['Bank transfer (automatic)', 'Credit card (automatic)']
SERVICE_COLUMNS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
CATEGORICAL_COLUMNS = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'PaymentMethod']
BINARY_COLUMNS = ['SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService']
NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_spend', 'service_diversity']

# =============================================================================
# BUSINESS RULES
# =============================================================================
MIN_BUSINESS_CLUSTERS = 3
MAX_BUSINESS_CLUSTERS = 6
HIGH_CHURN_THRESHOLD = 0.40
MEDIUM_CHURN_THRESHOLD = 0.25

# =============================================================================
# Ensure directories exist
# =============================================================================
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, DOCS_DIR, PRESENTATION_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

# Call on import
ensure_directories()