# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:16:27 2025

@author: user
"""

import numpy as np
import open3d as o3d
#import open3d as o3d
import matplotlib.pyplot as plt

#%% 2. Point Cloud Data Preparation

DATANAME = "cloud-garden.ply"
pcd = o3d.io.read_point_cloud("C:\\Users\\user\\Downloads\\cloud-garden/source\\cloud-garden\\cloud-garden.ply")
# Use relative path instead + DATANAME
#%% Data Pre-Processing
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)


#%% 3.1 Statistical Outlier Filter 

nn = 16
sd_multiplier = 2   #apply standard deviation multiplier to point cloud distribution

filtered_pcd = pcd.remove_statistical_outlier(nn,sd_multiplier)

outliers = pcd.select_by_index(filtered_pcd[1], invert=True)
outliers.paint_uniform_color([1, 0, 0])

filtered_pcd = filtered_pcd[0]

o3d.visualization.draw_geometries([filtered_pcd])

#%% 3.2 Voxel Downsampling 

voxel_size = 0.1

pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)
o3d.visualization.draw_geometries([pcd_downsampled])


#%% 3.3 Estimating normals 

nn_distance = np.mean(pcd.compute_nearest_neighbor_distance()) 

radius_normals = nn_distance*4 

pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16))

pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled,outliers])

#%% 4. Extracting and Setting Parameters
front = [ -0.020514513841629618, -0.95734267876571644, 0.28822586652749277 ]
lookat = [ -0.02186016887476061, -0.38136364006128343, 2.0003154805535668 ]
up = [ -0.027743388990222179, 0.28872061385378545, 0.95701134345588124 ]
zoom =  0.69999999999999996

pcd = pcd_downsampled
o3d.visualization.draw_geometries([pcd], zoom=zoom, front=front, lookat=lookat, up=up)


#%%  RANSAC Planar Segmentation

# Set distance threshold for RANSAC plane fitting
distance_threshold = 0.02

# Segment the largest plane in the point cloud
plane_model, inliers = pcd.segment_plane(
    distance_threshold=distance_threshold,
    ransac_n=3,
    num_iterations=1000)

# Unpack plane coefficients
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
print(f"Number of inliers: {len(inliers)}")

# Extract inlier and outlier point clouds
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

# Color the segmented plane and the rest of the cloud
inlier_cloud.paint_uniform_color([1.0, 0, 0])       # Red for plane
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])  # Gray for others

# Optional: Define camera view parameters for better visualization
view_params = {
    "zoom": 0.8,
    "front": [0.0, 0.0, -1.0],
    "lookat": [0.0, 0.0, 0.0],
    "up": [0.0, -1.0, 0.0]
}

# Visualize the result
o3d.visualization.draw_geometries(
    [inlier_cloud, outlier_cloud],
    zoom=view_params["zoom"],
    front=view_params["front"],
    lookat=view_params["lookat"],
    up=view_params["up"])

#%% 6. Multi-order RANSAC 
import open3d as o3d
import matplotlib.pyplot as plt

# Parameters
max_plane_idx = 6
pt_to_plane_dist = 0.02

# Optional: Downsample for performance
pcd = pcd.voxel_down_sample(voxel_size=0.01)

# Initialize containers
segment_models = {}
segments = {}
remaining_pcd = pcd

# Plane segmentation loop
for i in range(max_plane_idx):
    color = plt.get_cmap("tab20")(i)[:3]  # Extract RGB from RGBA
    model, inliers = remaining_pcd.segment_plane(
        distance_threshold=pt_to_plane_dist,
        ransac_n=3,
        num_iterations=1000
    )

    # Skip if too few inliers
    if len(inliers) < 50:
        print(f"Plane {i} has too few inliers, skipping.")
        continue

    # Store model and segment
    segment_models[i] = {
        "plane_model": model,
        "inliers": inliers,
        "normal": model[:3],
        "distance": model[3]
    }

    segments[i] = remaining_pcd.select_by_index(inliers)
    segments[i].paint_uniform_color(list(color))
    remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    print(f"Pass {i + 1}/{max_plane_idx} done.")

# Prepare visualization
geometries = list(segments.values())
if len(remaining_pcd.points) > 0:
    geometries.append(remaining_pcd)

o3d.visualization.draw_geometries(geometries)

        #%% DBSCAN sur rest 
import numpy as np
rest = pcd
labels = np.array(rest.cluster_dbscan(eps=0.5, min_points=5))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / max_label if max_label > 0 else 1)

colors[labels < 0] = 0 
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])





