from pathlib import Path
import numpy as np
from opensfm import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
from typing import Tuple, Dict, Any, List, Optional
images = Path('datasets/South-Building/images/')

outputs = Path('outputs/sfm/')
sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs / 'sfm_superpoint+superglue'
def extract_features(image: np.ndarray, config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)