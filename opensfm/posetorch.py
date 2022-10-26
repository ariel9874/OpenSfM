import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from pathlib import Path
import logging
import argparse
import time
import glob
import os
from typing import Tuple, Dict, Any, List, Optional
from matplotlib import pyplot as plt     
import cv2
import torch
from torch import nn
import numpy as np
from opensfm import context, pyfeatures
from opensfm import extract_features, matchs_features, reconstructions, visualization, pairs_from_exhaustive

# from diferents keypoints detectos and descriptors in pytorch match witch superglue

def posecalculator( images, config: Dict[str, Any])-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = Path('outputs/demo/')
    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sfm'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'
    poses = {}
    feature_conf =config["feature_type2"].upper()
    matcher_conf = matchs_features.confs['superglue']
    references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]
    extract_features.main(feature_conf, images, image_list=references, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    matchs_features.main(matcher_conf, sfm_pairs, features=features, matches=matches);
    return poses