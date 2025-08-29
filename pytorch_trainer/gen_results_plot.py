import argparse
import cv2
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import scipy.stats
import models

parser = argparse.ArgumentParser()
parser.add_argument("--load-pkl", action='store_true',
                    help="Load predictions for a cached pickle file or \
                        recompute from scratch by feeding the data through \
                        trained models")
args = parser.parse_args()


class Model(Enum):
    GroundTruth = "GroundTruth"
    Dropout = "Dropout"
    Ensemble = "Ensemble"
    Evidential = "Evidential"
    Gaussian = "Gaussian"
    Laplace = "Laplace"


save_dir = "noisy_pretrained_models"

