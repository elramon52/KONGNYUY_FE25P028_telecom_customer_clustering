# Telecom Customer Segmentation — EE637 ML Project

## Kongnyuy Raymond Afoni (FE25P028) | University of Buea | February 2026

---

## Project Description

This project implements a complete machine learning pipeline for segmenting telecom customers based on their usage patterns, demographics, and service subscriptions. Using unsupervised clustering algorithms (K-Means, Agglomerative, DBSCAN), the project discovers natural customer groupings that enable targeted marketing strategies and improved retention efforts.

### Key Features

- **Complete ML Pipeline**: From data loading to model deployment
- **Multiple Clustering Algorithms**: K-Means, Agglomerative, DBSCAN comparison
- **Business-Driven Model Selection**: Overrides mathematical optima when necessary
- **Comprehensive Validation**: Internal metrics + external validation via churn
- **Interactive Dashboard**: Real-time segment prediction with Streamlit

---

## Directory Structure

```
KONGNYUY_FE25P028_telecom_customer_clustering/
├── main.py                         # Orchestrates the entire pipeline
├── dashboard.py                     # Streamlit prediction app
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/
│   └── telecom_customer_dataset.csv # IBM telecom dataset
├── src/                             # Modular source code
│   ├── __init__.py
│   ├── config.py                    # Paths, random seeds, settings
│   ├── data_loader.py               # Load and inspect data
│   ├── splitter.py                  # Train/validation/test split
│   ├── eda.py                       # Exploratory Data Analysis
│   ├── preprocessing.py             # Feature engineering & scaling
│   ├── model_selection.py           # Algorithm evaluation
│   ├── training.py                  # Final model training
│   ├── profiling.py                 # Cluster characterization
│   └── utils.py                     # Helper functions
├── models/
│   ├── final_model.pkl              # Trained K-Means model
│   └── preprocessor.pkl             # Fitted preprocessing pipeline
├── reports/
│   ├── figures/                     # All generated plots (15 PNGs)
│   ├── cluster_profiles.csv         # Cluster statistics
│   └── business_actions.csv         # Per-segment recommendations

```

---

## Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.9+ | Core programming language |
| pandas | 1.5.0+ | Data manipulation |
| numpy | 1.23.0+ | Numerical computations |
| scikit-learn | 1.2.0+ | ML algorithms |
| scipy | 1.9.0+ | Statistical tests |
| matplotlib | 3.6.0+ | Plotting |
| seaborn | 0.12.0+ | Statistical visualizations |
| streamlit | 1.20.0+ | Interactive dashboard |


---

## Installation

```bash
# Clone or navigate to project directory
cd KONGNYUY_FE25P028_telecom_customer_clustering
# Create virtual environment
python -m venv venv
# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1: Run the Complete ML Pipeline

```bash
python main.py
```

This executes:
1. Data loading and inspection
2. Stratified train/validation/test split (60/20/20)
3. Exploratory Data Analysis (3 composite figures)
4. Data preprocessing and feature engineering
5. Model selection (K-Means, Agglomerative, DBSCAN)
6. Final model training and test evaluation
7. Cluster profiling and business action generation
8. Workflow and pipeline diagram generation

**Expected runtime**: 2-5 minutes depending on hardware.


### Step 2: Launch the Dashboard

```bash
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Then open your browser to:
- Local: http://localhost:8501
- Network: http://[your-ip]:8501

---

## Dashboard Features

The Streamlit dashboard provides:

1. **Prediction Tab**: Input customer details to predict their segment
2. **Segment Overview**: View cluster profiles and key metrics
3. **Feature Importance**: Understand which features drive segmentation
4. **All Plots**: Browse all 15 generated visualizations
5. **About**: Project information and methodology

---

## Generated Outputs

### Figures (15 total)

**EDA Composites (3)**:
- `eda_composite_1.png`: Demographics and tenure
- `eda_composite_2.png`: Service adoption and contracts
- `eda_composite_3.png`: Relationships and outliers

**Model Selection (5)**:
- `13_silhouette_vs_k_kmeans.png`
- `14_silhouette_vs_k_agglomerative.png`
- `15_dbscan_silhouette_heatmap.png`
- `16_elbow_method_kmeans.png`
- `17_algorithm_comparison.png`

**Test Evaluation (5)**:
- `18_churn_rate_by_cluster.png`
- `19_pca_projection_test.png`
- `20_cluster_centroids_heatmap.png`
- `21_silhouette_per_sample.png`
- `22_cluster_size_pie.png`

**Profiling (2)**:
- `23_radar_chart.png`
- `24_feature_importance_bar.png`

### Models

- `final_model.pkl`: Trained K-Means model
- `preprocessor.pkl`: Fitted preprocessing pipeline

### Reports

- `cluster_profiles.csv`: Cluster statistics and characteristics
- `business_actions.csv`: Marketing and retention strategies per segment

---

## Methodology Summary

### Data Splitting
- **Training**: 60% (4,225 records) — for fitting models
- **Validation**: 20% (1,409 records) — for model selection
- **Test**: 20% (1,409 records) — for final evaluation
- **Stratification**: Preserves 26.5% churn rate across splits

### Feature Engineering
Seven new features created:
1. `avg_monthly_spend`: Average monthly expenditure
2. `tenure_group`: Customer lifecycle stage (New/Mid/Loyal)
3. `service_diversity`: Count of add-on services
4. `contract_commitment`: Ordinal contract length
5. `auto_payment`: Automatic payment flag
6. `high_value_flag`: Premium customer indicator
7. `family_status`: Household composition

### Model Selection
- **Algorithms**: K-Means, Agglomerative, DBSCAN
- **Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- **Business Override**: K=2 rejected (too trivial), K>6 rejected (too granular)
- **Selected**: K-Means with K=3

### Validation
- **Internal**: Silhouette = 0.42, DB Index = 0.85, CH Index = 2850
- **External**: Chi-square p < 0.001, Cramer's V = 0.25

---

## Key Findings

1. **Three distinct customer segments** identified with clear business interpretations
2. **Strong external validation**: Significant association between clusters and churn
3. **Top discriminating features**: Contract type, tenure, monthly charges
4. **Actionable insights**: Each segment has tailored marketing and retention strategies

---

## License

This project is created for academic purposes as part of the Masters of Engineering program at the University of Buea.

---

## Acknowledgments

- IBM for providing the Telecom Customer Churn dataset
- Instructor Kinge Mbeke Theophane Osee for guidance
- University of Buea for academic support
