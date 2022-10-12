"""Tools to extract features."""

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

logger: logging.Logger = logging.getLogger(__name__)


class SemanticData:
    segmentation: np.ndarray
    instances: Optional[np.ndarray]
    labels: List[Dict[str, Any]]

    def __init__(
        self,
        segmentation: np.ndarray,
        instances: Optional[np.ndarray],
        labels: List[Dict[str, Any]],
        
    ):
        self.segmentation = segmentation
        self.instances = instances
        self.labels = labels

    def has_instances(self) -> bool:
        return self.instances is not None

    def mask(self, mask: np.ndarray) -> "SemanticData":
        try:
            segmentation = self.segmentation[mask]
            instances = self.instances
            if instances is not None:
                instances = instances[mask]
        except IndexError:
            logger.error(
                f"Invalid mask array of dtype {mask.dtype}, shape {mask.shape}: {mask}"
            )
            raise

        return SemanticData(segmentation, instances, self.labels)


class FeaturesData:

    points: np.ndarray
    descriptors: Optional[np.ndarray]
    colors: np.ndarray
    semantic: Optional[SemanticData]

    FEATURES_VERSION: int = 3
    FEATURES_HEADER: str = "OPENSFM_FEATURES_VERSION"

    def __init__(
        self,
        points: np.ndarray,
        descriptors: Optional[np.ndarray],
        colors: np.ndarray,
        semantic: Optional[SemanticData],
    ):
        self.points = points
        self.descriptors = descriptors
        self.colors = colors
        self.semantic = semantic

    def get_segmentation(self) -> Optional[np.ndarray]:
        semantic = self.semantic
        if not semantic:
            return None
        if semantic.segmentation is not None:
            return semantic.segmentation
        return None

    def has_instances(self) -> bool:
        semantic = self.semantic
        if not semantic:
            return False
        return semantic.instances is not None

    def mask(self, mask: np.ndarray) -> "FeaturesData":
        if self.semantic:
            masked_semantic = self.semantic.mask(mask)
        else:
            masked_semantic = None
        return FeaturesData(
            self.points[mask],
            self.descriptors[mask] if self.descriptors is not None else None,
            self.colors[mask],
            masked_semantic,
        )

    def save(self, fileobject: Any, config: Dict[str, Any]):
        """Save features from file (path like or file object like)"""
        feature_type = config["feature_type"]
        if (
            (
                feature_type == "AKAZE"
                and config["akaze_descriptor"] in ["MLDB_UPRIGHT", "MLDB"]
            )
            or (feature_type == "HAHOG" and config["hahog_normalize_to_uchar"])
            or (feature_type == "ORB")
        ):
            feature_data_type = np.uint8
        else:
            feature_data_type = np.float32
        descriptors = self.descriptors
        if descriptors is None:
            raise RuntimeError("No descriptors found, cannot save features data.")
        semantic = self.semantic
        if semantic:
            instances = semantic.instances
            np.savez_compressed(
                fileobject,
                points=self.points.astype(np.float32),
                descriptors=descriptors.astype(feature_data_type),
                colors=self.colors,
                segmentations=semantic.segmentation.astype(np.uint8),
                instances=instances.astype(np.int16) if instances is not None else [],
                segmentation_labels=np.array(semantic.labels).astype(np.str),
                OPENSFM_FEATURES_VERSION=self.FEATURES_VERSION,
            )
        else:
            np.savez_compressed(
                fileobject,
                points=self.points.astype(np.float32),
                descriptors=descriptors.astype(feature_data_type),
                colors=self.colors,
                segmentations=[],
                instances=[],
                segmentation_labels=[],
                OPENSFM_FEATURES_VERSION=self.FEATURES_VERSION,
            )

    @classmethod
    def from_file(cls, fileobject: Any, config: Dict[str, Any]) -> "FeaturesData":
        """Load features from file (path like or file object like)"""
        s = np.load(fileobject, allow_pickle=False)
        version = cls._features_file_version(s)
        return getattr(cls, "_from_file_v%d" % version)(s, config)

    @classmethod
    def _features_file_version(cls, obj: Dict[str, Any]) -> int:
        """Retrieve features file version. Return 0 if none"""
        if cls.FEATURES_HEADER in obj:
            return obj[cls.FEATURES_HEADER]
        else:
            return 0

    @classmethod
    def _from_file_v0(
        cls, data: Dict[str, np.ndarray], config: Dict[str, Any]
    ) -> "FeaturesData":
        """Base version of features file

        Scale (desc[2]) set to reprojection_error_sd by default (legacy behaviour)
        """
        feature_type = config["feature_type"]
        if feature_type == "HAHOG" and config["hahog_normalize_to_uchar"]:
            descriptors = data["descriptors"].astype(np.float32)
        else:
            descriptors = data["descriptors"]
        points = data["points"]
        points[:, 2:3] = config["reprojection_error_sd"]
        return FeaturesData(points, descriptors, data["colors"].astype(float), None)

    @classmethod
    def _from_file_v1(
        cls, data: Dict[str, np.ndarray], config: Dict[str, Any]
    ) -> "FeaturesData":
        """Version 1 of features file

        Scale is not properly set higher in the pipeline, default is gone.
        """
        feature_type = config["feature_type"]
        if feature_type == "HAHOG" and config["hahog_normalize_to_uchar"]:
            descriptors = data["descriptors"].astype(np.float32)
        else:
            descriptors = data["descriptors"]
        return FeaturesData(
            data["points"], descriptors, data["colors"].astype(float), None
        )

    @classmethod
    def _from_file_v2(
        cls,
        data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> "FeaturesData":
        """
        Version 2 of features file

        Added segmentation, instances and segmentation labels. This version has been introduced at
        e5da878bea455a1e4beac938cb30b796acfe3c8c, but has been superseded by version 3 as this version
        uses 'allow_pickle=True' which isn't safe (RCE vulnerability)
        """
        feature_type = config["feature_type"]
        if feature_type == "HAHOG" and config["hahog_normalize_to_uchar"]:
            descriptors = data["descriptors"].astype(np.float32)
        else:
            descriptors = data["descriptors"]

        # luckily, because os lazy loading, we can still load 'segmentations' and 'instances' ...
        pickle_message = (
            "Cannot load {} as these were generated with "
            "version 2 which isn't supported anymore because of RCE vulnerablity."
            "Please consider re-extracting features data for this dataset"
        )
        try:
            has_segmentation = (data["segmentations"] != None).all()
            has_instances = (data["instances"] != None).all()
        except ValueError:
            logger.warning(pickle_message.format("segmentations and instances"))
            has_segmentation, has_instances = False, False

        # ... whereas 'labels' can't be loaded anymore, as it is a plain 'list' object. Not an
        # issue since these labels are used for description only and not actual filtering.
        try:
            labels = data["segmentation_labels"]
        except ValueError:
            logger.warning(pickle_message.format("labels"))
            labels = []

        if has_segmentation or has_instances:
            semantic_data = SemanticData(
                data["segmentations"] if has_segmentation else None,
                data["instances"] if has_instances else None,
                labels,
            )
        else:
            semantic_data = None
        return FeaturesData(
            data["points"], descriptors, data["colors"].astype(float), semantic_data
        )

    @classmethod
    def _from_file_v3(
        cls,
        data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> "FeaturesData":
        """
        Version 3 of features file

        Same as version 2, except that
        """
        feature_type = config["feature_type"]
        if feature_type == "HAHOG" and config["hahog_normalize_to_uchar"]:
            descriptors = data["descriptors"].astype(np.float32)
        else:
            descriptors = data["descriptors"]

        has_segmentation = len(data["segmentations"]) > 0
        has_instances = len(data["instances"]) > 0

        if has_segmentation or has_instances:
            semantic_data = SemanticData(
                data["segmentations"] if has_segmentation else None,
                data["instances"] if has_instances else None,
                data["segmentation_labels"],
            )
        else:
            semantic_data = None
        return FeaturesData(
            data["points"], descriptors, data["colors"].astype(float), semantic_data
        )
"""
def simple_nms(scores, nms_radius: int):
    
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }

        """      
class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc


class SuperPointFrontend(object):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
               cuda=True):
    self.name = 'SuperPoint'
    self.cuda = True
    self.nms_dist = nms_dist
    self.conf_thresh = conf_thresh
    self.nn_thresh = nn_thresh # L2 descriptor distance for good match.
    self.cell = 8 # Size of each output cell. Keep this fixed.
    self.border_remove = 4 # Remove points this close to the border.

    # Load the network in inference mode.
    self.net = SuperPointNet()
    if cuda:
      # Train on GPU, deploy on GPU.
      self.net.load_state_dict(torch.load(weights_path))
      self.net = self.net.cuda()
    else:
      # Train on GPU, deploy on CPU.
      self.net.load_state_dict(torch.load(weights_path,
                               map_location=lambda storage, loc: storage))
    self.net.eval()

  def nms_fast(self, in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run(self, img):
    """ Process a numpy image to extract points and descriptors.
    Input
      img - HxW numpy float32 input image in range [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - 256xN numpy array of corresponding unit normalized descriptors.
      heatmap - HxW numpy heatmap in range [0,1] of point confidences.
      """
    assert img.ndim == 2, 'Image must be grayscale.'
    assert img.dtype == np.float32, 'Image must be float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    if self.cuda:
      inp = inp.cuda()
    # Forward pass of network.
    outs = self.net.forward(inp)
    semi, coarse_desc = outs[0], outs[1]
    # Convert pytorch -> numpy.
    semi = semi.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      if self.cuda:
        samp_pts = samp_pts.cuda()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return pts, desc, heatmap

def resized_image(image: np.ndarray, max_size: int) -> np.ndarray:
    """Resize image to feature_process_size."""
    h, w = image.shape[:2]
    size = max(w, h)
    if 0 < max_size < size:
        dsize = w * max_size // size, h * max_size // size
        return cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_AREA)
    else:
        return image


def root_feature(desc: np.ndarray, l2_normalization: bool = False) -> np.ndarray:
    if l2_normalization:
        s2 = np.linalg.norm(desc, axis=1)
        desc = (desc.T / s2).T
    s = np.sum(desc, 1)
    desc = np.sqrt(desc.T / s).T
    return desc


def root_feature_surf(
    desc: np.ndarray, l2_normalization: bool = False, partial: bool = False
) -> np.ndarray:
    """
    Experimental square root mapping of surf-like feature, only work for 64-dim surf now
    """
    if desc.shape[1] == 64:
        if l2_normalization:
            s2 = np.linalg.norm(desc, axis=1)
            desc = (desc.T / s2).T
        if partial:
            ii = np.array([i for i in range(64) if (i % 4 == 2 or i % 4 == 3)])
        else:
            ii = np.arange(64)
        desc_sub = np.abs(desc[:, ii])
        desc_sub_sign = np.sign(desc[:, ii])
        # s_sub = np.sum(desc_sub, 1)  # This partial normalization gives slightly better results for AKAZE surf
        s_sub = np.sum(np.abs(desc), 1)
        desc_sub = np.sqrt(desc_sub.T / s_sub).T
        desc[:, ii] = desc_sub * desc_sub_sign
    return desc


def normalized_image_coordinates(
    pixel_coords: np.ndarray, width: int, height: int
) -> np.ndarray:
    size = max(width, height)
    p = np.empty((len(pixel_coords), 2))
    p[:, 0] = (pixel_coords[:, 0] + 0.5 - width / 2.0) / size
    p[:, 1] = (pixel_coords[:, 1] + 0.5 - height / 2.0) / size
    return p


def denormalized_image_coordinates(
    norm_coords: np.ndarray, width: int, height: int
) -> np.ndarray:
    size = max(width, height)
    p = np.empty((len(norm_coords), 2))
    p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
    p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0
    return p


def normalize_features(
    points: np.ndarray, desc: np.ndarray, colors: np.ndarray, width: int, height: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,]:
    """Normalize feature coordinates and size."""
    points[:, :2] = normalized_image_coordinates(points[:, :2], width, height)
    points[:, 2:3] /= max(width, height)
    return points, desc, colors


def _in_mask(point: np.ndarray, width: int, height: int, mask: np.ndarray) -> bool:
    """Check if a point is inside a binary mask."""
    u = mask.shape[1] * (point[0] + 0.5) / width
    v = mask.shape[0] * (point[1] + 0.5) / height
    return mask[int(v), int(u)] != 0


def extract_features_sift(
    image: np.ndarray, config: Dict[str, Any], features_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    sift_edge_threshold = config["sift_edge_threshold"]
    sift_peak_threshold = float(config["sift_peak_threshold"])
    # SIFT support is in cv2 main from version 4.4.0
    for i in range(30):
        for j in range(30):
            print("* ", end="")
            print()
    if context.OPENCV44 or context.OPENCV5:
        # OpenCV versions concerned /** 3.4.11, >= 4.4.0 **/  ==> Sift became free since March 2020
        detector = cv2.SIFT_create(
            edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold
        )
        descriptor = detector
    elif context.OPENCV3 or context.OPENCV4:
        try:
            # OpenCV versions concerned /** 3.2.x, 3.3.x, 3.4.0, 3.4.1, 3.4.2, 3.4.10, 4.3.0, 4.4.0 **/
            detector = cv2.xfeatures2d.SIFT_create(
                edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold
            )
        except AttributeError as ae:
            # OpenCV versions concerned /** 3.4.3, 3.4.4, 3.4.5, 3.4.6, 3.4.7, 3.4.8, 3.4.9, 4.0.x, 4.1.x, 4.2.x **/
            if "no attribute 'xfeatures2d'" in str(ae):
                logger.error(
                    "OpenCV Contrib modules are required to extract SIFT features"
                )
            raise
        descriptor = detector
    else:
        detector = cv2.FeatureDetector_create("SIFT")
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        detector.setDouble("edgeThreshold", sift_edge_threshold)
    while True:
        logger.debug("Computing sift with threshold {0}".format(sift_peak_threshold))
        t = time.time()
        # SIFT support is in cv2 main from version 4.4.0
        if context.OPENCV44 or context.OPENCV5:
            detector = cv2.SIFT_create(
                edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold
            )
        elif context.OPENCV3:
            detector = cv2.xfeatures2d.SIFT_create(
                edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold
            )
        else:
            detector.setDouble("contrastThreshold", sift_peak_threshold)
        points = detector.detect(image)
        logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))
        if len(points) < features_count and sift_peak_threshold > 0.0001:
            sift_peak_threshold = (sift_peak_threshold * 2) / 3
            logger.debug("reducing threshold")
        else:
            logger.debug("done")
            break
    points, desc = descriptor.compute(image, points)

    if desc is not None:
        if config["feature_root"]:
            desc = root_feature(desc)
        points = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
    else:
        points = np.array(np.zeros((0, 3)))
        desc = np.array(np.zeros((0, 3)))
    return points, desc
def extract_features_superpoint(
    image: np.ndarray, config: Dict[str, Any], features_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    sift_edge_threshold = config["sift_edge_threshold"]
    sift_peak_threshold = float(config["sift_peak_threshold"])
    # SIFT support is in cv2 main from version 4.4.0
    weights_paths='superpoint_v1.pth'
    nms_dists=0
    conf_threshs=0.09
    nn_threshs=0.4
    for i in range(30):
        for j in range(30):
            print("* ", end="")
            print()
    fe = SuperPointFrontend(weights_path=weights_paths,
                          nms_dist=nms_dists,
                          conf_thresh=conf_threshs,
                          nn_thresh=nn_threshs,
                          cuda=True)
    image = (image.astype('float32') / 255.)
    points, desc, heatmap = fe.run(image)

    if desc is not None:
        if config["feature_root"]:
            desc = root_feature(desc)
        points = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
    else:
        points = np.array(np.zeros((0, 3)))
        desc = np.array(np.zeros((0, 3)))
    return points, desc
def extract_features_popsift(
    image: np.ndarray, config: Dict[str, Any], features_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    from opensfm import pypopsift

    sift_edge_threshold = float(config["sift_edge_threshold"])
    sift_peak_threshold = float(config["sift_peak_threshold"])

    points, desc = pypopsift.popsift(image, peak_threshold=sift_peak_threshold,
                                edge_threshold=sift_edge_threshold,
                                target_num_features=features_count,
                                use_root=bool(config["feature_root"]))

    return points, desc

def extract_features_surf(
    image: np.ndarray, config: Dict[str, Any], features_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    surf_hessian_threshold = config["surf_hessian_threshold"]
    if context.OPENCV3:
        try:
            detector = cv2.xfeatures2d.SURF_create()
        except AttributeError as ae:
            if "no attribute 'xfeatures2d'" in str(ae):
                logger.error(
                    "OpenCV Contrib modules are required to extract SURF features"
                )
            raise
        descriptor = detector
        detector.setHessianThreshold(surf_hessian_threshold)
        detector.setNOctaves(config["surf_n_octaves"])
        detector.setNOctaveLayers(config["surf_n_octavelayers"])
        detector.setUpright(config["surf_upright"])
    else:
        detector = cv2.FeatureDetector_create("SURF")
        descriptor = cv2.DescriptorExtractor_create("SURF")
        detector.setDouble("hessianThreshold", surf_hessian_threshold)
        detector.setDouble("nOctaves", config["surf_n_octaves"])
        detector.setDouble("nOctaveLayers", config["surf_n_octavelayers"])
        detector.setInt("upright", config["surf_upright"])

    while True:
        logger.debug("Computing surf with threshold {0}".format(surf_hessian_threshold))
        t = time.time()
        if context.OPENCV3:
            detector.setHessianThreshold(surf_hessian_threshold)
        else:
            detector.setDouble(
                "hessianThreshold", surf_hessian_threshold
            )  # default: 0.04
        points = detector.detect(image)
        logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))
        if len(points) < features_count and surf_hessian_threshold > 0.0001:
            surf_hessian_threshold = (surf_hessian_threshold * 2) / 3
            logger.debug("reducing threshold")
        else:
            logger.debug("done")
            break

    points, desc = descriptor.compute(image, points)

    if desc is not None:
        if config["feature_root"]:
            desc = root_feature(desc)
        points = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
    else:
        points = np.array(np.zeros((0, 3)))
        desc = np.array(np.zeros((0, 3)))
    return points, desc


def akaze_descriptor_type(name: str) -> pyfeatures.AkazeDescriptorType:
    d = pyfeatures.AkazeDescriptorType.__dict__
    if name in d:
        return d[name]
    else:
        logger.debug("Wrong akaze descriptor type")
        return d["MSURF"]


def extract_features_akaze(
    image: np.ndarray, config: Dict[str, Any], features_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    options = pyfeatures.AKAZEOptions()
    options.omax = config["akaze_omax"]
    akaze_descriptor_name = config["akaze_descriptor"]
    options.descriptor = akaze_descriptor_type(akaze_descriptor_name)
    options.descriptor_size = config["akaze_descriptor_size"]
    options.descriptor_channels = config["akaze_descriptor_channels"]
    options.dthreshold = config["akaze_dthreshold"]
    options.kcontrast_percentile = config["akaze_kcontrast_percentile"]
    options.use_isotropic_diffusion = config["akaze_use_isotropic_diffusion"]
    options.target_num_features = features_count
    options.use_adaptive_suppression = config["feature_use_adaptive_suppression"]

    logger.debug("Computing AKAZE with threshold {0}".format(options.dthreshold))
    t = time.time()
    points, desc = pyfeatures.akaze(image, options)
    logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))

    if config["feature_root"]:
        if akaze_descriptor_name in ["SURF_UPRIGHT", "MSURF_UPRIGHT"]:
            desc = root_feature_surf(desc, partial=True)
        elif akaze_descriptor_name in ["SURF", "MSURF"]:
            desc = root_feature_surf(desc, partial=False)
    points = points.astype(float)
    return points, desc


def extract_features_hahog(
    image: np.ndarray, config: Dict[str, Any], features_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    t = time.time()
    points, desc = pyfeatures.hahog(
        image.astype(np.float32) / 255,  # VlFeat expects pixel values between 0, 1
        peak_threshold=config["hahog_peak_threshold"],
        edge_threshold=config["hahog_edge_threshold"],
        target_num_features=features_count,
    )

    if config["feature_root"]:
        desc = np.sqrt(desc)
        uchar_scaling = 362  # x * 512 < 256  =>  sqrt(x) * 362 < 256
    else:
        uchar_scaling = 512

    if config["hahog_normalize_to_uchar"]:
        desc = (uchar_scaling * desc).clip(0, 255).round()

    logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))
    return points, desc


def extract_features_orb(
    image: np.ndarray, config: Dict[str, Any], features_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    if context.OPENCV3:
        detector = cv2.ORB_create(nfeatures=features_count)
        descriptor = detector
    else:
        detector = cv2.FeatureDetector_create("ORB")
        descriptor = cv2.DescriptorExtractor_create("ORB")
        detector.setDouble("nFeatures", features_count)

    logger.debug("Computing ORB")
    t = time.time()
    points = detector.detect(image)

    points, desc = descriptor.compute(image, points)
    if desc is not None:
        points = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
    else:
        points = np.array(np.zeros((0, 3)))
        desc = np.array(np.zeros((0, 3)))

    logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))
    return points, desc


def extract_features(
    image: np.ndarray, config: Dict[str, Any], is_panorama: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect features in a color or gray-scale image.

    The type of feature detected is determined by the ``feature_type``
    config option.

    The coordinates of the detected points are returned in normalized
    image coordinates.

    Parameters:
        - image: a color image with shape (h, w, 3) or
                 gray-scale image with (h, w) or (h, w, 1)
        - config: the configuration structure
        - is_panorama : if True, alternate settings are used for feature count and extraction size.

    Returns:
        tuple:
        - points: ``x``, ``y``, ``size`` and ``angle`` for each feature
        - descriptors: the descriptor of each feature
        - colors: the color of the center of each feature
    """
    extraction_size = (
        config["feature_process_size_panorama"]
        if is_panorama
        else config["feature_process_size"]
    )
    features_count = (
        config["feature_min_frames_panorama"]
        if is_panorama
        else config["feature_min_frames"]
    )

    assert len(image.shape) == 3 or len(image.shape) == 2
    image = resized_image(image, extraction_size)
    if len(image.shape) == 2:  # convert (h, w) to (h, w, 1)
        image = np.expand_dims(image, axis=2)
    # convert color to gray-scale if necessary
    if image.shape[2] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    keypoints = None
    feature_type = config["feature_type"].upper()
    if feature_type == "SIFT2":
        points, desc = extract_features_sift(image_gray, config, features_count)
    elif feature_type == "SURF":
        points, desc = extract_features_surf(image_gray, config, features_count)
    elif feature_type == "AKAZE":
        points, desc = extract_features_akaze(image_gray, config, features_count)
    elif feature_type == "HAHOG":
        points, desc = extract_features_hahog(image_gray, config, features_count)
    elif feature_type == "ORB":
        points, desc = extract_features_orb(image_gray, config, features_count)
    elif feature_type == 'SIFT_GPU':
        points, desc = extract_features_popsift(image_gray, config, features_count)
    elif feature_type=="SIFT":
        points,desc=extract_features_superpoint(image_gray, config, features_count)
    else:
        raise ValueError(
            "Unknown feature type " "(must be SURF, SIFT, AKAZE, HAHOG, SIFT_GPU or ORB)"
        )

    xs = points[:, 0].round().astype(int)
    ys = points[:, 1].round().astype(int)
    colors = image[ys, xs]
    if image.shape[2] == 1:
        colors = np.repeat(colors, 3).reshape((-1, 3))

    if keypoints is not None:
        return normalize_features(points, desc, colors,
                                  image.shape[1], image.shape[0]), keypoints

    return normalize_features(points, desc, colors, image.shape[1], image.shape[0])


def build_flann_index(descriptors: np.ndarray, config: Dict[str, Any]) -> Any:
    # FLANN_INDEX_LINEAR = 0
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_KMEANS = 2
    # FLANN_INDEX_COMPOSITE = 3
    # FLANN_INDEX_KDTREE_SINGLE = 4
    # FLANN_INDEX_HIERARCHICAL = 5

    if descriptors.dtype.type is np.float32:
        algorithm_type = config["flann_algorithm"].upper()
        if algorithm_type == "KMEANS":
            FLANN_INDEX_METHOD = FLANN_INDEX_KMEANS
        elif algorithm_type == "KDTREE":
            FLANN_INDEX_METHOD = FLANN_INDEX_KDTREE
        else:
            raise ValueError("Unknown flann algorithm type " "must be KMEANS, KDTREE")
        flann_params = {
            "algorithm": FLANN_INDEX_METHOD,
            "branching": config["flann_branching"],
            "iterations": config["flann_iterations"],
            "tree": config["flann_tree"],
        }
    else:
        raise ValueError(
            "FLANN isn't supported for binary features because of poor-performance. Use BRUTEFORCE instead."
        )

    return context.flann_Index(descriptors, flann_params)
