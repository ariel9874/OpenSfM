import datetime
import enum
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
    reconstruction_helpers as helpers,
    rig,
    tracking,
    types,
)
from opensfm.align import align_reconstruction, apply_similarity
from opensfm.context import current_memory_usage, parallel_map
from opensfm.dataset_base import DataSetBase


logger: logging.Logger = logging.getLogger(__name__)
from opensfm import reconstruction
from opensfm.actions import reconstruct

from . import command
import argparse
from opensfm.dataset import DataSet
def denormalized_image_coordinates(
    norm_coords: np.ndarray, width: int, height: int
) -> np.ndarray:
    size = max(width, height)
    p = np.empty((len(norm_coords), 2))
    p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
    p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0
    return p

def normalized_image_coordinates(
    pixel_coords: np.ndarray, width: int, height: int
) -> np.ndarray:
    size = max(width, height)
    p = np.empty((len(pixel_coords), 2))
    p[:, 0] = (pixel_coords[:, 0] + 0.5 - width / 2.0) / size
    p[:, 1] = (pixel_coords[:, 1] + 0.5 - height / 2.0) / size
    return p
    
class Command(command.CommandBase):
    name = "ariel"
    help = "Compute the gcps"
    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        #read gcps
        tracks_manager = dataset.load_tracks_manager()
        images = tracks_manager.get_shot_ids()
        dataset.init_reference(images)
        gcps = dataset.load_ground_control_points()
        remaining_images = set(images)
        #read reconstruction
        reconstruction = dataset.load_reconstruction()
        for e in reconstruction:
            for a in remaining_images:
                if a in remaining_images:
                        shot = remaining_images[a]
                        print("a")
                        print(shot)
            remaining_images-= set(e.shots)
            for a in remaining_images:
                if a in remaining_images:
                        shot = remaining_images[a]
                        print("a")
                        print(shot)        
            """print("shot01")
            print(e.get_shot("01.jpg"))
            axs=e.get_shot("01.jpg")
            print("mapa get shot 01")
            print(e.map.get_shot("01.jpg"))
            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            print(e.map)
            mapa=e.map
            f=mapa.get_shot("01.jpg")
            t=mapa.get_landmarks()
            for aa in t:
                print("landmark")
                print(aa)
            print("landmarks")
            print(t)
            print("shot")
            shotts=e.shots
            for shott in shots:
                print("shot2")
                print(shott)
            print(f)
            x=1205.0
            y=1052.0"""
            """width=axs.width#aqui da error
            height=axs.height
            f=[x,y]
            pos=normalized_image_coordinates(f, width, height)
            print("pos")
            print(e.get_shot("01.jpg").pose)
            print("rotation")
            
            print(e.get_shot("01.jpg").pose.get_rotation_matrix())
            print("origin")
            print(e.get_shot("01.jpg").pose.get_origin())
            print("cameras")
            print(e.get_cameras())"""
        
        for gcp in gcps:
            print("gcp",gcp)

            for obs in gcp.observations:
                print("obs",obs)
                x= obs.projection
                print("x",x)
                print("shot_id",obs.shot_id)
                cords=data.image_size("01.jpg")
                f=denormalized_image_coordinates(x,cords[0],cords[1])
                print("f",f)
        camera_priors = data.load_camera_models()
        rig_camera_priors = data.load_rig_cameras()
        for i in gcps:
            print("hello")
            logging.info("Extracting EXIF obfor {}".format(i.observations))
            print("Lla_vec ori")
            print(*i.lla_vec)
            print("Lla_vec topo")
            print(e.reference.to_topocentric(*i.lla_vec))
            
            for es in i.observations:
                print("es")
                print(es)
                print("vars")
                print("projection")
                print(es.projection)
                print(es.projection[0])
                print(es.projection[1])
                shot_id=es.shot_id
                if shot_id in remaining_images:
                    shot = remaining_images[shot_id]
                    print("pase")
                #faa=denormalized_image_coordinates(es.projection, shot.shape[1], shot.shape[0])
                #print(faa)
                print("values")
                print(es.values())
                if es.geodetic_measurement:
                    print("geodetic")
                    print(es.geodetic_measurement.to_dict())
                
                print("shot_id")
                print(es.shot_id)
        for reconstruction in reconstructions:
            print("reconstruction",reconstruction)
            print("reconstruction.shots",reconstruction.shots)
            print("reconstruction.points",reconstruction.points)
            for i in reconstruction.shots:
                print("reconstruction.shots",i)
            print("reconstruction.cameras",reconstruction.cameras)
            print("reconstruction.rigs",reconstruction.rig_instances)
            print("reconstruction.rig_cameras",reconstruction.rig_cameras)
            print("//////////////////////////////////////////////////////////////////////////////")
            print("//////////////////////////////////////////////////////////////////////////////")
            print("//////////////////////////////////////////////////////////////////////////////")
            print("//////////////////////////////////////////////////////////////////////////////")
            for i in reconstruction.points:
                print("reconstruccion",i)
            for camera in reconstruction.cameras.values():
                print("camera",camera)  
                print("camerajonson",camera_to_json(camera))
                print("//////////////////////////////////////////////////////////////////////////////")
            # Extract cameras biases
            for camera_id, bias in reconstruction.biases.items():
                print("camera_id",camera_id)
                print("bias",bias)
                print("bias_to_json",bias_to_json(bias))
            # Extract rig models
            for rig_camera in reconstruction.rig_cameras.values():
                print("rig_camera",rig_camera)
                print("rig_camera_to_json",rig_camera_to_json(rig_camera))
            for rig_instance in reconstruction.rig_instances.values():
                print("rig_instance",rig_instance)
                print("rig_instance_to_json",rig_instance_to_json(rig_instance))
            # Extract shots
            for shot in reconstruction.shots.values():
                print("shot",shot)
                print("shot.id",shot.id)
                print("shot_to_json",shot_to_json(shot))
            # Extract points
            for point in reconstruction.points.values():
                print("point",point)    
                print("point_to_json",point_to_json(point))
                print("point.coordinates",point.coordinates)
            if hasattr(reconstruction, "pano_shots"):
                if len(reconstruction.pano_shots) > 0:
                    for shot in reconstruction.pano_shots.values():
                        print("shot",shot)
                        print("shot_to_json",shot_to_json(shot))
            
            reconstruction.map
            print("GCPs",gcps)
            for gcp in gcps:
                print("gcp",gcp.observations)
            #print("out",gcp_geopositional_error(gcps, reconstruction))
            #print("out REPOJECT",reproject_gcps(gcps, reconstruction, reproj_threshold=10))
            print(reconstruction.map)
            for shot_id in reconstruction.shots:
                shot = reconstruction.get_shot(shot_id)
                #for point in shot.get_valid_landmarks():
                 #   obs = shot.get_landmark_observation(point)
                  #  print("shot id",shot.id)
                   #print("point id ",point.id)
                  #  print("obs point ",obs.point)
                   # print("obs scale ",obs.scale)
                    #####nota fin del día TypeError: 'opensfm.pymap.LandmarkView' object is not callable y revisar run_ba.py
        return gcps
    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        pass
        
    
      