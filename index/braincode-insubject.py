"""Three independently optimized within-subject EEG decoding procedures.

The accompanying notebook embeds this complete source; no companion file is
required on Kaggle. This is an experimental pipeline, not a performance claim.
"""

# %% Configuration and reproducibility
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import random
import re
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, asdict, replace
from fractions import Fraction
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.special import softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import oas
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, log_loss, classification_report)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

VERSION = "insubject-1.0"
SOURCE_SHA256 = "development"  # replaced by the notebook builder
CLASSES = np.arange(4)
CLASS_NAMES = ("Up", "Down", "Right", "Left")
BANDS = ((2., 4.), (4., 8.), (8., 13.), (13., 20.), (20., 30.), (30., 45.))
WINDOWS = {"full": ((1., 3.5),), "late": ((1.5, 3.5),),
           "early": ((1., 3.),), "two_views": ((1., 3.), (1.5, 3.5))}


@dataclass
class ExperimentConfig:
    data_root: str = "/kaggle/input/datasets/truthisneverlinear/inner-speech-recognition/inner-speech-recognition/derivatives"
    output_root: str = "/kaggle/working/inner_speech_optimized"
    profile: str = "smoke"
    subject_ids: tuple | None = (1,)
    channel_counts: tuple = (128,)
    pipelines: tuple = ("classical", "eegnet", "labram")
    cv_seeds: tuple = (42,)
    outer_folds: int = 2
    inner_folds: int = 2
    ensemble_seeds: tuple = (0,)
    max_epochs: int = 3
    min_epochs: int = 1
    patience: int = 2
    scratch_batch_size: int = 32
    labram_batch_size: int = 8
    amp: bool = True
    device: str = "auto"
    cpu_threads: int = 2
    resume: bool = True
    expected_sfreq: float = 256.
    labram_repo: str = "braindecode/labram-pretrained"
    labram_revision: str = "0563b6c626e7b40d9a36653b763715db94d945d7"
    labram_offline: bool = False
    label_permutation_seed: int | None = None  # diagnostic only; labeled in every output
    # Optional explicit candidate lists. None selects the profile's documented list.
    classical_candidates: tuple | None = None
    eegnet_candidates: tuple | None = None
    labram_candidates: tuple | None = None

    def validate(self):
        if self.profile not in ("smoke", "pilot", "full"):
            raise ValueError("profile must be smoke, pilot, or full")
        for field in ("channel_counts", "pipelines", "cv_seeds", "ensemble_seeds"):
            value = getattr(self, field)
            if not isinstance(value, (tuple, list)) or not value or len(set(value)) != len(value):
                raise ValueError(f"{field} must be a nonempty tuple with unique entries; use (128,), not (128)")
        if not set(self.channel_counts) <= {16, 32, 64, 128}:
            raise ValueError("Use channel counts 128, 64, 32, or 16")
        if not set(self.pipelines) <= {"classical", "eegnet", "labram"}:
            raise ValueError("Unknown pipeline")