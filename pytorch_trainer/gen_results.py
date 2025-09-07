import argparse
from collections import defaultdict
import cv2
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import scipy.stats
parser = argparse.ArgumentParser()
parser.add_argument("--load-pkl", action='store_true',
                    help="Load predictions for a cached pickle file or \
                        recompute from scratch by feeding the data through \
                        trained models")
args = parser.parse_args()


class Model(Enum):
    GroundTruth = "GroundTruth"
    Gaussian = "Gaussian"
    Laplace = "Laplace"
    Generalized = "GeneralizedGaussian"

save_dir = ""
output_dir = ""

trained_models = {

}

def compute_predictions()

