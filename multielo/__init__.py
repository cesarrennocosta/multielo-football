"""
multielo-football: Multi-Dimensional Elo Ratings and Poisson Prediction Hierarchy for International Football
"""

from .datasets import load_dataset, download_dataset
from .ratings import compute_ratings
from .predict import predict
from .metrics import compute_rps, compute_esd, compute_aic
from .models import get_model_specs, GLM_TAXONOMY

__version__ = "0.1.0"
__author__ = "César Rennó-Costa, László Csató"

__all__ = [
    "load_dataset",
    "download_dataset",
    "compute_ratings",
    "predict",
    "compute_rps",
    "compute_esd",
    "compute_aic",
    "get_model_specs",
    "GLM_TAXONOMY",
]
