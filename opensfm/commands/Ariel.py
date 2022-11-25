import datetime
import enum
import logging
import math
import os
#import open3d as o3d
from opensfm.actions import compute_depthmaps
from . import command
from opensfm import dense
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterable
import sys
sys.path.append("../OpenSfM/annotation_gui_gcp/lib/")
from annotation_gui_gcp.lib.gcp_manager import GroundControlPointManager
import cv2
import numpy as np
from pyparsing import null_debug_action
from opensfm import (
    log,
    matching,
    multiview,
    pybundle,
    pygeometry,
    pymap,
    pysfm,
    align,
    reconstruction_helpers as helpers,
    rig,
    tracking,
    types,
)
from opensfm.align import align_reconstruction, apply_similarity
from opensfm.context import current_memory_usage, parallel_map
from opensfm.dataset_base import DataSet_base
logger: logging.Logger = logging.getLogger(__name__)
from opensfm import reconstruction
from opensfm.actions import reconstruct
import argparse
from opensfm.dataset import DataSet
def denormalized_image_coordinates(
    norm_coords: np.ndarray, width: int, height: int
) -> np.ndarray:
    size = max(width, height)
    p = np.empty((len(norm_coords), 2))
    p[0] = norm_coords[0] * size - 0.5 + width / 2.0
    p[1] = norm_coords[1] * size - 0.5 + height / 2.0
    return p

def normalized_image_coordinates(
    pixel_coords: np.ndarray, width: int, height: int
) -> np.ndarray:
    size = max(width, height)
    p = np.empty((len(pixel_coords), 2))
    p[ 0] = (pixel_coords[0] + 0.5 - width / 2.0) / size
    p[1] = (pixel_coords[1] + 0.5 - height / 2.0) / size
    return p

def triangulate_gcps(gcps: List[pymap.GroundControlPoint], reconstruction: types.Reconstruction):
    coords = []
    for gcp in gcps:
        res = multiview.triangulate_gcp(
            gcp,
            reconstruction.shots,
            reproj_threshold=1,
            min_ray_angle_degrees=0.1,
        )
        coords.append(res)
    return coords
def gcp_geopositional_error(gcps: List[pymap.GroundControlPoint], reconstruction: types.Reconstruction):
    coords_reconstruction = triangulate_gcps(gcps, reconstruction)
    print("coords_reconstuction",coords_reconstruction)
    out = {}
    for ix, gcp in enumerate(gcps):
        print("gcp",gcp.id)
        expected = reconstruction.reference.to_topocentric(*gcp.lla_vec) if gcp.lla else None
        print("expected",expected)
        print("lla_vec",*gcp.lla_vec if gcp.lla else None)
        triangulated = (
            coords_reconstruction[ix] if coords_reconstruction[ix] is not None else None
        )
        print("triangulated",triangulated)

        if expected is not None and triangulated is not None:
            error = np.linalg.norm(expected - triangulated)
            out[gcp.id] = {
                "expected_xyz": list(expected),
                "triangulated_xyz": list(triangulated),
                "expected_lla": reconstruction.reference.to_lla(*expected),
                "triangulated_lla": reconstruction.reference.to_lla(*triangulated),
                "error": float(error),
            }

            # Compute the metric error, ignoring altitude
            lat, lon, _alt = out[gcp.id]["expected_lla"]
            expected_xy = reconstruction.reference.to_topocentric(lat, lon, 0)
            lat, lon, _alt = out[gcp.id]["triangulated_lla"]
            triangulated_xy = reconstruction.reference.to_topocentric(lat, lon, 0)
            out[gcp.id]["error_planar"] = np.linalg.norm(
                np.array(expected_xy) - np.array(triangulated_xy)
            )
        else:
            out[gcp.id] = {"error": np.nan, "error_planar": np.nan}

    return out  
def camera_from_json(key: str, obj: Dict[str, Any]) -> pygeometry.Camera:
    """
    Read camera from a json object
    """
    camera = None
    pt = obj.get("projection_type", "perspective")
    if pt == "perspective":
        camera = pygeometry.Camera.create_perspective(
            obj["focal"], obj.get("k1", 0.0), obj.get("k2", 0.0)
        )
    elif pt == "brown":
        camera = pygeometry.Camera.create_brown(
            obj["focal_x"],
            obj["focal_y"] / obj["focal_x"],
            np.array([obj.get("c_x", 0.0), obj.get("c_y", 0.0)]),
            np.array(
                [
                    obj.get("k1", 0.0),
                    obj.get("k2", 0.0),
                    obj.get("k3", 0.0),
                    obj.get("p1", 0.0),
                    obj.get("p2", 0.0),
                ]
            ),
        )
    elif pt == "fisheye":
        camera = pygeometry.Camera.create_fisheye(
            obj["focal"], obj.get("k1", 0.0), obj.get("k2", 0.0)
        )
    elif pt == "fisheye_opencv":
        camera = pygeometry.Camera.create_fisheye_opencv(
            obj["focal_x"],
            obj["focal_y"] / obj["focal_x"],
            np.array([obj.get("c_x", 0.0), obj.get("c_y", 0.0)]),
            np.array(
                [
                    obj.get("k1", 0.0),
                    obj.get("k2", 0.0),
                    obj.get("k3", 0.0),
                    obj.get("k4", 0.0),
                ]
            ),
        )
    elif pt == "fisheye62":
        camera = pygeometry.Camera.create_fisheye62(
            obj["focal_x"],
            obj["focal_y"] / obj["focal_x"],
            np.array([obj.get("c_x", 0.0), obj.get("c_y", 0.0)]),
            np.array(
                [
                    obj.get("k1", 0.0),
                    obj.get("k2", 0.0),
                    obj.get("k3", 0.0),
                    obj.get("k4", 0.0),
                    obj.get("k5", 0.0),
                    obj.get("k6", 0.0),
                    obj.get("p1", 0.0),
                    obj.get("p2", 0.0),
                ]
            ),
        )
    elif pt == "fisheye624":
        camera = pygeometry.Camera.create_fisheye624(
            obj["focal_x"],
            obj["focal_y"] / obj["focal_x"],
            np.array([obj.get("c_x", 0.0), obj.get("c_y", 0.0)]),
            np.array(
                [
                    obj.get("k1", 0.0),
                    obj.get("k2", 0.0),
                    obj.get("k3", 0.0),
                    obj.get("k4", 0.0),
                    obj.get("k5", 0.0),
                    obj.get("k6", 0.0),
                    obj.get("p1", 0.0),
                    obj.get("p2", 0.0),
                    obj.get("s0", 0.0),
                    obj.get("s1", 0.0),
                    obj.get("s2", 0.0),
                    obj.get("s3", 0.0),
                ]
            ),
        )
    elif pt == "radial":
        camera = pygeometry.Camera.create_radial(
            obj["focal_x"],
            obj["focal_y"] / obj["focal_x"],
            np.array([obj.get("c_x", 0.0), obj.get("c_y", 0.0)]),
            np.array(
                [
                    obj.get("k1", 0.0),
                    obj.get("k2", 0.0),
                ]
            ),
        )
    elif pt == "simple_radial":
        camera = pygeometry.Camera.create_simple_radial(
            obj["focal_x"],
            obj["focal_y"] / obj["focal_x"],
            np.array([obj.get("c_x", 0.0), obj.get("c_y", 0.0)]),
            obj.get("k1", 0.0),
        )
    elif pt == "dual":
        camera = pygeometry.Camera.create_dual(
            obj.get("transition", 0.5),
            obj["focal"],
            obj.get("k1", 0.0),
            obj.get("k2", 0.0),
        )
    elif pygeometry.Camera.is_panorama(pt):
        camera = pygeometry.Camera.create_spherical()
    else:
        raise NotImplementedError
    camera.id = key
    camera.width = int(obj.get("width", 0))
    camera.height = int(obj.get("height", 0))
    return camera

 
def reproject_gcps(gcps: List[pymap.GroundControlPoint], reconstruction: types.Reconstruction, reproj_threshold):
    output = {}
    for gcp in gcps:
        point = multiview.triangulate_gcp(
            gcp,
            reconstruction.shots,
            reproj_threshold=reproj_threshold,
            min_ray_angle_degrees=0.1,
        )
        print("point",point)
        output[gcp.id] = {}
        print("gcp.id",gcp.id)
        n_obs = len(gcp.observations)
        print("n_obs",n_obs)
        if point is None:
            logger.info(f"Could not triangulate {gcp.id} with {n_obs} annotations")
            continue
        for observation in gcp.observations:
            lat, lon, alt = reconstruction.reference.to_lla(*point)
            print("lat,lon,alt",lat,lon,alt)
            output[gcp.id][observation.shot_id] = {"lla": [lat, lon, alt], "error": 0}
            print("observation.shot_id",observation.shot_id not in reconstruction.shots)
            if observation.shot_id not in reconstruction.shots:
                continue
            shot = reconstruction.shots[observation.shot_id]
            print("shot",shot)
            reproj = shot.project(point)
            print("reproj",reproj)
            print("observation.projection",observation.projection)
            error = np.linalg.norm(reproj - observation.projection)
            print("error",error)
            output[gcp.id][observation.shot_id].update(
                {
                    "error": error,
                    "reprojection": [reproj[0], reproj[1]],
                }
            )
    return output 

def shot_to_json(shot: pymap.Shot) -> Dict[str, Any]:
    """
    Write shot to a json object
    """
    obj: Dict[str, Any] = {
        "rotation": list(shot.pose.rotation),
        "translation": list(shot.pose.translation),
        "camera": shot.camera.id,
    }

    if shot.metadata is not None:
        obj.update(pymap_metadata_to_json(shot.metadata))
    if shot.mesh is not None:
        obj["vertices"] = [list(vertice) for vertice in shot.mesh.vertices]
        obj["faces"] = [list(face) for face in shot.mesh.faces]
    if hasattr(shot, "scale"):
        obj["scale"] = shot.scale
    if hasattr(shot, "covariance"):
        obj["covariance"] = shot.covariance.tolist()
    if hasattr(shot, "merge_cc"):
        obj["merge_cc"] = shot.merge_cc
    return obj


def rig_instance_to_json(rig_instance: pymap.RigInstance) -> Dict[str, Any]:
    """
    Write a rig instance to a json object
    """
    return {
        "translation": list(rig_instance.pose.translation),
        "rotation": list(rig_instance.pose.rotation),
        "rig_camera_ids": rig_instance.rig_camera_ids,
    }


def rig_camera_to_json(rig_camera: pymap.RigCamera) -> Dict[str, Any]:
    """
    Write a rig camera to a json object
    """
    obj = {
        "rotation": list(rig_camera.pose.rotation),
        "translation": list(rig_camera.pose.translation),
    }
    return obj


def pymap_metadata_to_json(metadata: pymap.ShotMeasurements) -> Dict[str, Any]:
    obj = {}
    if metadata.orientation.has_value:
        obj["orientation"] = metadata.orientation.value
    if metadata.capture_time.has_value:
        obj["capture_time"] = metadata.capture_time.value
    if metadata.gps_accuracy.has_value:
        obj["gps_dop"] = metadata.gps_accuracy.value
    if metadata.gps_position.has_value:
        obj["gps_position"] = list(metadata.gps_position.value)
    if metadata.gravity_down.has_value:
        obj["gravity_down"] = list(metadata.gravity_down.value)
    if metadata.compass_angle.has_value and metadata.compass_accuracy.has_value:
        obj["compass"] = {
            "angle": metadata.compass_angle.value,
            "accuracy": metadata.compass_accuracy.value,
        }
    else:
        if metadata.compass_angle.has_value:
            obj["compass"] = {"angle": metadata.compass_angle.value}
        elif metadata.compass_accuracy.has_value:
            obj["compass"] = {"accuracy": metadata.compass_accuracy.value}
    if metadata.sequence_key.has_value:
        obj["skey"] = metadata.sequence_key.value
    return obj


def json_to_pymap_metadata(obj: Dict[str, Any]) -> pymap.ShotMeasurements:
    metadata = pymap.ShotMeasurements()
    if obj.get("orientation") is not None:
        metadata.orientation.value = obj.get("orientation")
    if obj.get("capture_time") is not None:
        metadata.capture_time.value = obj.get("capture_time")
    if obj.get("gps_dop") is not None:
        metadata.gps_accuracy.value = obj.get("gps_dop")
    if obj.get("gps_position") is not None:
        metadata.gps_position.value = obj.get("gps_position")
    if obj.get("skey") is not None:
        metadata.sequence_key.value = obj.get("skey")
    if obj.get("gravity_down") is not None:
        metadata.gravity_down.value = obj.get("gravity_down")
    if obj.get("compass") is not None:
        compass = obj.get("compass")
        if "angle" in compass:
            metadata.compass_angle.value = compass["angle"]
        if "accuracy" in compass:
            metadata.compass_accuracy.value = compass["accuracy"]
    return metadata


def point_to_json(point: pymap.Landmark) -> Dict[str, Any]:
    """
    Write a point to a json object
    """
    return {
        "color": list(point.color.astype(float)),
        "coordinates": list(point.coordinates),
    }

def camera_to_json(camera) -> Dict[str, Any]:
    """
    Write camera to a json object
    """
    if camera.projection_type == "perspective":
        return {
            "projection_type": camera.projection_type,
            "width": camera.width,
            "height": camera.height,
            "focal": camera.focal,
            "k1": camera.k1,
            "k2": camera.k2,
        }
    elif camera.projection_type == "brown":
        return {
            "projection_type": camera.projection_type,
            "width": camera.width,
            "height": camera.height,
            "focal_x": camera.focal,
            "focal_y": camera.focal * camera.aspect_ratio,
            "c_x": camera.principal_point[0],
            "c_y": camera.principal_point[1],
            "k1": camera.k1,
            "k2": camera.k2,
            "p1": camera.p1,
            "p2": camera.p2,
            "k3": camera.k3,
        }
    elif camera.projection_type == "fisheye":
        return {
            "projection_type": camera.projection_type,
            "width": camera.width,
            "height": camera.height,
            "focal": camera.focal,
            "k1": camera.k1,
            "k2": camera.k2,
        }
    elif camera.projection_type == "fisheye_opencv":
        return {
            "projection_type": camera.projection_type,
            "width": camera.width,
            "height": camera.height,
            "focal_x": camera.focal,
            "focal_y": camera.focal * camera.aspect_ratio,
            "c_x": camera.principal_point[0],
            "c_y": camera.principal_point[1],
            "k1": camera.k1,
            "k2": camera.k2,
            "k3": camera.k3,
            "k4": camera.k4,
        }
    elif camera.projection_type == "fisheye62":
        return {
            "projection_type": camera.projection_type,
            "width": camera.width,
            "height": camera.height,
            "focal_x": camera.focal,
            "focal_y": camera.focal * camera.aspect_ratio,
            "c_x": camera.principal_point[0],
            "c_y": camera.principal_point[1],
            "k1": camera.k1,
            "k2": camera.k2,
            "k3": camera.k3,
            "k4": camera.k4,
            "k5": camera.k5,
            "k6": camera.k6,
            "p1": camera.p1,
            "p2": camera.p2,
        }
    elif camera.projection_type == "fisheye624":
        return {
            "projection_type": camera.projection_type,
            "width": camera.width,
            "height": camera.height,
            "focal_x": camera.focal,
            "focal_y": camera.focal * camera.aspect_ratio,
            "c_x": camera.principal_point[0],
            "c_y": camera.principal_point[1],
            "k1": camera.k1,
            "k2": camera.k2,
            "k3": camera.k3,
            "k4": camera.k4,
            "k5": camera.k5,
            "k6": camera.k6,
            "p1": camera.p1,
            "p2": camera.p2,
            "s0": camera.s0,
            "s1": camera.s1,
            "s2": camera.s2,
            "s3": camera.s3,
        }
    elif camera.projection_type == "simple_radial":
        return {
            "projection_type": camera.projection_type,
            "width": camera.width,
            "height": camera.height,
            "focal_x": camera.focal,
            "focal_y": camera.focal * camera.aspect_ratio,
            "c_x": camera.principal_point[0],
            "c_y": camera.principal_point[1],
            "k1": camera.k1,
        }
    elif camera.projection_type == "radial":
        return {
            "projection_type": camera.projection_type,
            "width": camera.width,
            "height": camera.height,
            "focal_x": camera.focal,
            "focal_y": camera.focal * camera.aspect_ratio,
            "c_x": camera.principal_point[0],
            "c_y": camera.principal_point[1],
            "k1": camera.k1,
            "k2": camera.k2,
        }
    elif camera.projection_type == "dual":
        return {
            "projection_type": camera.projection_type,
            "width": camera.width,
            "height": camera.height,
            "focal": camera.focal,
            "k1": camera.k1,
            "k2": camera.k2,
            "transition": camera.transition,
        }
    elif pygeometry.Camera.is_panorama(camera.projection_type):
        return {
            "projection_type": camera.projection_type,
            "width": camera.width,
            "height": camera.height,
        }
    else:
        raise NotImplementedError

def reconstruction_to_json(reconstruction: types.Reconstruction) -> Dict[str, Any]:
    """
    Write a reconstruction to a json object
    """
    obj = {"cameras": {}, "shots": {}, "points": {}, "biases": {}}

    # Extract cameras
    for camera in reconstruction.cameras.values():
        obj["cameras"][camera.id] = camera_to_json(camera)

    # Extract cameras biases
    for camera_id, bias in reconstruction.biases.items():
        obj["biases"][camera_id] = bias_to_json(bias)

    # Extract rig models
    if len(reconstruction.rig_cameras):
        obj["rig_cameras"] = {}
    for rig_camera in reconstruction.rig_cameras.values():
        obj["rig_cameras"][rig_camera.id] = rig_camera_to_json(rig_camera)
    if len(reconstruction.rig_instances):
        obj["rig_instances"] = {}
    for rig_instance in reconstruction.rig_instances.values():
        obj["rig_instances"][rig_instance.id] = rig_instance_to_json(rig_instance)

    # Extract shots
    for shot in reconstruction.shots.values():
        obj["shots"][shot.id] = shot_to_json(shot)

    # Extract points
    for point in reconstruction.points.values():
        obj["points"][point.id] = point_to_json(point)

    # Extract pano_shots
    if hasattr(reconstruction, "pano_shots"):
        if len(reconstruction.pano_shots) > 0:
            obj["pano_shots"] = {}
            for shot in reconstruction.pano_shots.values():
                obj["pano_shots"][shot.id] = shot_to_json(shot)

    # Extract reference topocentric frame
    if reconstruction.reference:
        ref = reconstruction.reference
        obj["reference_lla"] = {
            "latitude": ref.lat,
            "longitude": ref.lon,
            "altitude": ref.alt,
        }

    return obj


def reconstructions_to_json(
    reconstructions: Iterable[types.Reconstruction],
) -> List[Dict[str, Any]]:
    """
    Write all reconstructions to a json object
    """
    return [reconstruction_to_json(i) for i in reconstructions]


def cameras_to_json(cameras: Dict[str, pygeometry.Camera]) -> Dict[str, Dict[str, Any]]:
    """
    Write cameras to a json object
    """
    obj = {}
    for camera in cameras.values():
        obj[camera.id] = camera_to_json(camera)
    return obj


def bias_to_json(bias: pygeometry.Similarity) -> Dict[str, Any]:
    return {
        "rotation": list(bias.rotation),
        "translation": list(bias.translation),
        "scale": bias.scale,
    }


def rig_cameras_to_json(
    rig_cameras: Dict[str, pymap.RigCamera]
) -> Dict[str, Dict[str, Any]]:
    """
    Write rig cameras to a json object
    """
    obj = {}
    for rig_camera in rig_cameras.values():
        obj[rig_camera.id] = rig_camera_to_json(rig_camera)
    return obj

class Command(command.CommandBase):
    name = "ariel"
    help = "Compute the gcps"
    def run_impl(self, data: DataSet, args: argparse.Namespace) -> None:
        report = {}
        config = data.config
        tracks_manager = data.load_tracks_manager()
        images = tracks_manager.get_shot_ids()
        udata_path = os.path.join(data.data_path, args.subfolder)
        udata = data.UndistortedDataSet(data, udata_path)
        #nube = o3d.io.read_point_cloud('./opensfm/undistorted/openmvs/scene_dense_dense_filtered.ply')

        
        data.init_reference(images)

        remaining_images = set(images)
        gcps = data.load_ground_control_points()
        common_tracks = tracking.all_common_tracks_with_features(tracks_manager)
        reconstructions = []
        reconstructions=data.load_reconstruction()
        for reconstruction in reconstructions:
            shot= reconstruction.shots["01.jpg"]
            print("shot",shot)
            p1 = shot.pose.get_origin()
            print("p1",p1)
            x=1024
            print("x",x)
            y=1024
            print("y",y)
            p2 = shot.camera.pixel_to_normalized_coordinates([x, y])
            print("p2",p2)
            bearing = shot.camera.pixel_bearing(p2)
            print("bearing",bearing)
            scale = 1 / bearing[2]
            print("scale",scale)
            bearing = scale * bearing
            print("bearing scale",bearing)
            p2 = shot.pose.inverse().transform(bearing)
            print("p2 inverse",p2)
            dMap, plane, score, nghbr, nghbrs = udata.load_raw_depthmap(shot.id)
            points=np.asarray(nube.points)
            print("points",points)
            res = np.linalg.norm(np.cross(p2-p1, p1-points), axis=1)/np.linalg.norm(p2-p1)
            print("res",res)
            #points = np.asarray(nube.points) #point cloud
            #res = np.linalg.norm(np.cross(p2-p1, p1-points), axis=1)/np.linalg.norm(p2-p1)
        return 1
    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--subfolder",
            help="undistorted subfolder where to load and store data",
            default="undistorted",
        )
        parser.add_argument(
            "--interactive",
            help="plot results as they are being computed",
            action="store_true",
        )
        pass
        