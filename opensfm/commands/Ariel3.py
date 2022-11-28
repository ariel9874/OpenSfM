from opensfm.actions import compute_depthmaps
import os
from opensfm import dataset
from opensfm import dense
from opensfm.dataset import DataSet
from . import command
import argparse
from opensfm.dataset import DataSet
import open3d as o3d
import cv2
import numpy as np
from opensfm.geo import (ecef_from_lla, topocentric_from_lla)
from opensfm import (
    log,
    io,
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
    types,)
def estimate_depth(map, x, y, steps):
    step = 1
    depth = map[x][y]
    print('point depth')
    print(depth)
    if depth != 0.0:
        while(step <= steps):
            sum = 0
            devisor = 0
            for i in range(x-step, x+step+1):
                devisor += 1
                for j in range(y-step, y+step+1):
                    sum = sum + map[i][j]
                    devisor += 1
            average = sum / devisor
            if depth != 0:
                depth = (depth + average) / 2
            else:
                depth = average
            
            print('average depth')
            print(depth)
            step += 1
    return depth
class Command(command.CommandBase):
    name = "ariel"
    help = "Compute depthmap2"

    def run_impl(self, data: DataSet, args: argparse.Namespace) -> None:
        udata_path = os.path.join(data.data_path, args.subfolder)
        udataset = dataset.UndistortedDataSet(data, udata_path, io_handler=data.io_handler)
        data.config["interactive"] = args.interactive
        reconstructions = udataset.load_undistorted_reconstruction()
        tracks_manager = udataset.load_undistorted_tracks_manager()
        dense.compute_depthmaps(udataset, tracks_manager, reconstructions[0])
        
        report = {}
        config = data.config
        images = tracks_manager.get_shot_ids()
        data.init_reference(images)

        remaining_images = set(images)
        gcps = data.load_ground_control_points()
        print("path",udata_path)
        common_tracks = tracking.all_common_tracks_with_features(tracks_manager)
        distancias = []
        diff = []
        for reconstruction in reconstructions:
            
            """print("shot",shot)
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
            points, normals, colors, labels = udataset.load_point_cloud(shot.id)
            #points=np.asarray(points)
            print("points",points)
            res = np.linalg.norm(np.cross(p2-p1, p1-points), axis=1)/np.linalg.norm(p2-p1)
            print("res",res)"""
            nube=io.reconstruction_to_ply(reconstruction, tracks_manager, False, False, False)
            print("nube",nube)
            nube=o3d.io.read_point_cloud(data.data_path+'/reconstruction.ply')
            
            x=1024
            print("x",x)
            y=1024
            print("y",y)
            distancias=[]
            anterior=[]
            diferenciasgps=[]
            diferenciasunidades=[]
            contador=0
            valores=[]
            for i in images:
                for b in range(0,3000):
                    shot= reconstruction.shots[i]
                    print("shot",shot.id)
                    pose = shot.pose
                    cam = shot.camera
                    x=b
                    y=b
                    print("x",x)
                    print("y",y)
                    p1 = shot.pose.get_origin()
                    print("p1",p1)
                    p2 = cam.pixel_to_normalized_coordinates([x, y])
                    print("p2",p2)
                    bearing = cam.pixel_bearing(p2)
                    print("bearing",bearing)
                    scale = 1 / bearing[2]
                    print("scale",scale)
                    bearing = scale * bearing
                    print("bearing scale",bearing)
                    p2 = pose.inverse().transform(bearing)
                    print("p2 inverse",p2)
                    gps_world =reconstruction.reference.to_lla(*p2)
                    print("gps_world",gps_world)
                    valores.append(gps_world)
                    points = np.asarray(nube.points) #point cloud
                    print("points",points)
                    res = np.linalg.norm(np.cross(p2-p1, p1-points), axis=1)/np.linalg.norm(p2-p1)
                    #print("np.cross",np.cross(p2-p1, p1-points))
                    #print("np.linalg.norm",np.linalg.norm(np.cross(p2-p1, p1-points), axis=1))
                    #print("np.linalg.norm(p2-p1)",np.linalg.norm(p2-p1))
                    #print("res",res)
                    a=nube.points[np.argmin(res)]
                    print("pos inter",a)
                    distancias.append(a)
                    valores.append(shot.id)
                    valores.append(a)
                    gps_world = reconstruction.reference.to_lla(*a)
                    valores.append(gps_world)
                    print("gps_world inter",gps_world)
                    valores.append(x)
                    valores.append(y)
                    r = shot.pose.get_rotation_matrix().T
                    p2 = shot.pose.inverse().transform(bearing)
                    f=r.dot(p2)
                    print("pos bearing escala",f)
                    valores.append(f)
                    gps_world = reconstruction.reference.to_lla(*f)
                    valores.append(gps_world)
                    print('\nGPS Worldyo bearing escala',gps_world)
                    p2 = cam.pixel_to_normalized_coordinates([x, y])
                    bearing = cam.pixel_bearing(p2)
                    r = shot.pose.get_rotation_matrix().T
                    f=r.dot(bearing)
                    print("pos bearing",f)
                    valores.append(f)
                    gps_world = reconstruction.reference.to_lla(*f)
                    print('\nGPS Worldyo bearing',gps_world)
                    valores.append(gps_world)
                    p1 = shot.pose.get_origin()
                    f=r.dot(p1)
                    print("pos origin",f)
                    gps_world = reconstruction.reference.to_lla(*f)
                    print('\nGPS Worldyo origin',gps_world) 
                    valores.append(f)
                    valores.append(gps_world)

        #for i in valores:
          #  if i in images:
              #  print("###############################################################################################")
             #   print("###############################################################################################")
           #     print("###############################################################################################")
            #    print("###############################################################################################")
          #  print(i)
                
            """for i in range(500,530):
                p1 = shot.pose.get_origin()
                print("p1",p1)
                x=i
                print("x",x)
                y=i
                print("y",y)
                ori=[x,y]
                distancias.append([x,y])
                camera = shot.camera
                pt2d = camera.pixel_to_normalized_coordinates((x, y))
                print("pt2d",pt2d)
                # bearing
                bearing = camera.pixel_bearing(pt2d)
                print('\nBearing')
                print(bearing)
                # bearing /= bearing[2]

                # Camera Dimensions
                print('\nCamera Dimensions')
                print(f'{camera.width} x {camera.width}')

                # Load depthmap dataset
                dMap, plane, score, nghbr, nghbrs = udataset.load_raw_depthmap(shot.id)
                print('\nDepthmap Size')
                print(f'{dMap[0].size} x {dMap[1].size}')

                print('\nMax of camera width/height') 
                m = max(camera.width, camera.height)
                print(m)

                print('\nScaling for X/Y')
                scaleX = dMap[0].size / m
                scaleY = dMap[1].size / m
                print(f'{scaleX} / {scaleY}')

                print('\nDepth Coordinates')
                dx = int(round((scaleX * float(x)), 1))
                dy = int(round((scaleY * float(y)), 1))
                print(f'{dx} x {dy}')

                print('\nPoint Depth')
                depth = dMap[dx][dy]
                if (depth == 0.0):
                    depth = estimate_depth(dMap, dx, dy, 2)
                print(depth)
                
                # adjust depth
                pt3d = bearing * depth
                print('\nPoint 3D (bearing * depth)')
                print(pt3d)
                xxxx = shot.project(pt3d)
                pt2D_px = camera.normalized_to_pixel_coordinates(xxxx)
                print('\nPixel Reverse Projection')
                print(pt2D_px)
                distancias.append([pt2D_px,depth])
                print("distancias",distancias)
                diff.append(ori-pt2D_px)
                print("diff",diff)
                    # this point is in the camera coordinate system
                pt3d_world = shot.pose.inverse().transform(pt3d)
                print('\nPoint 3D World')
                print(pt3d_world)

                # gps coords
                gps_world = reconstruction.reference.to_lla(*pt3d_world)
                print('\nGPS World')
                print(gps_world)

                # ECEF From LLA
                ecef = ecef_from_lla(*gps_world)
                print('\nECEF')
                print(ecef)

                # topocentric
                topo = topocentric_from_lla(gps_world[0], gps_world[1], gps_world[2], 0, 0, 0)
                print('\nTopocentric')
                print(topo)
                ############################################################################################################
                pose = shot.pose
                print("pose",pose)
                cam = shot.camera
                print("cam",cam)
                pt2D = cam.pixel_to_normalized_coordinates(pt2D_px)
                print("pt2D",pt2D)
                bearing = cam.pixel_bearing(pt2D)
                print("bearing",bearing)
                t3D_world = pose.inverse().transform(bearing)
                print("t3D_world",t3D_world)
                p1 = shot.pose.get_origin()
                print("p1",p1)
                p2 = cam.pixel_to_normalized_coordinates([x, y])
                print("p2",p2)
                bearing = cam.pixel_bearing(p2)
                print("bearing",bearing)
                scale = 1 / bearing[2]
                print("scale",scale)
                bearing = scale * bearing
                print("bearing scale",bearing)
                p2 = pose.inverse().transform(bearing)
                print("p2 inverse",p2)
                points = np.asarray(nube.points) #point cloud
                print("points",points)
                res = np.linalg.norm(np.cross(p2-p1, p1-points), axis=1)/np.linalg.norm(p2-p1)
                print("res",res)"""

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
