"""
Utility functions for the Telecom Customer Segmentation project.
Includes plotting helpers, save/load functions, and other helpers.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .config import (
    FIGURES_DIR, DPI, COLOR_PALETTE, MODEL_PATH, PREPROCESSOR_PATH,
    STUDENT_NAME, STUDENT_ID, PROGRAM, INSTITUTION, COURSE, INSTRUCTOR, DATE
)

# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def setup_plot_style():
    """Set up consistent plot style for all visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9


def save_figure(fig: plt.Figure, filename: str, figures_dir: Path = FIGURES_DIR) -> str:
    """
    Save figure to the figures directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the file (with or without extension)
        figures_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    setup_plot_style()
    if not filename.endswith('.png'):
        filename += '.png'
    
    filepath = figures_dir / filename
    fig.tight_layout()
    fig.savefig(filepath, format='png', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return str(filepath)


def create_color_gradient(n_colors: int, base_color: str = None) -> List[str]:
    """Create a color gradient from the base palette."""
    if base_color:
        colors = [base_color] * n_colors
    else:
        colors = COLOR_PALETTE[:n_colors]
    return colors


# =============================================================================
# SAVE/LOAD UTILITIES
# =============================================================================

def save_model(model: Any, filepath: Path = MODEL_PATH) -> str:
    """Save model to disk using joblib."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    return str(filepath)


def load_model(filepath: Path = MODEL_PATH) -> Any:
    """Load model from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_preprocessor(preprocessor: Any, filepath: Path = PREPROCESSOR_PATH) -> str:
    """Save preprocessor to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(preprocessor, f)
    return str(filepath)


def load_preprocessor(filepath: Path = PREPROCESSOR_PATH) -> Any:
    """Load preprocessor from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def compute_confidence_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wilson score interval for binomial proportion.
    
    Args:
        successes: Number of successes
        n: Total number of trials
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    if n == 0:
        return (0.0, 0.0)
    
    p = successes / n
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    
    return (max(0, centre - margin), min(1, centre + margin))


def cramers_v(confusion_matrix: np.ndarray) -> float:
    """
    Compute Cramer's V statistic for categorical-categorical association.
    
    Args:
        confusion_matrix: Contingency table
        
    Returns:
        Cramer's V value
    """
    from scipy import stats
    
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    # Corrected phi2
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# =============================================================================
# PRINT UTILITIES
# =============================================================================

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_section(title: str):
    """Print a section divider."""
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def print_project_identity():
    """Print project identity header."""
    print("=" * 80)
    print("            TELECOM CUSTOMER SEGMENTATION ML PROJECT".center(80))
    print("=" * 80)
    print(f"Student: {STUDENT_NAME} ({STUDENT_ID})")
    print(f"Program: {PROGRAM}")
    print(f"Institution: {INSTITUTION}")
    print(f"Course: {COURSE}")
    print(f"Instructor: {INSTRUCTOR}")
    print(f"Date: {DATE}")
    print("=" * 80)
