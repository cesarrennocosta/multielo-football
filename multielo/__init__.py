"""
multielo-football: Multi-Dimensional Elo Ratings and Poisson Prediction Hierarchy for International Football
"""

from .datasets import load_dataset, download_dataset
from .data_builder import build_balanced_learning_dataset, get_balanced_learning_dataset
from .ratings import compute_ratings
from .predict import predict
from .metrics import compute_rps, compute_esd, compute_aic, evaluate_5cv, evaluate_aics, evaluate_model, compute_metric_zscores
from .models import get_model_specs, GLM_TAXONOMY, train_model, TrainedModel, build_design_matrix

__version__ = "0.2.39"
__author__ = "César Rennó-Costa, László Csató"

__all__ = [
    "load_dataset",
    "download_dataset",
    "build_balanced_learning_dataset",
    "get_balanced_learning_dataset",
    "compute_ratings",
    "predict",
    "train_model",
    "TrainedModel",
    "build_design_matrix",
    "compute_rps",
    "compute_esd",
    "compute_aic",
    "evaluate_5cv",
    "evaluate_aics",
    "evaluate_model",
    "compute_metric_zscores",
    "get_model_specs",
    "GLM_TAXONOMY",
]
