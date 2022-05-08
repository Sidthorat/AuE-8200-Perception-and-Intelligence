# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:04:08 2022

@author: siddh
"""
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini',dataroot='C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes',verbose=True)
nusc.list_scenes()
my_scene = nusc.scene[0]
my_scene
first_sample_token = my_scene['first_sample_token']
nusc.render_sample(first_sample_token)
my_sample = nusc.get('sample', first_sample_token)
my_sample
nusc.list_sample(my_sample['token'])
my_sample['data']
sensor = 'CAM_FRONT'
cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
cam_front_data
nusc.render_sample_data(cam_front_data['token'])
sensor = 'CAM_BACK'
cam_back_data = nusc.get('sample_data', my_sample['data'][sensor])
cam_back_data
sensor = 'CAM_BACK_LEFT'
cam_back_left_data = nusc.get('sample_data', my_sample['data'][sensor])
cam_back_left_data
nusc.render_sample_data(cam_back_data['token'])
sensor = 'CAM_BACK_LEFT'
cam_back_left_data = nusc.get('sample_data', my_sample['data'][sensor])
cam_back_left_data
sensor = 'CAM_BACK_RIGHT'
cam_back_right_data = nusc.get('sample_data', my_sample['data'][sensor])
cam_back_right_data
nusc.render_sample_data(cam_back_right_data['token'])
sensor = 'CAM_FRONT_LEFT'
cam_front_left_data = nusc.get('sample_data', my_sample['data'][sensor])
cam_front_left_data
nusc.render_sample_data(cam_front_left_data['token'])
sensor = 'CAM_FRONT_RIGHT'
cam_front_right_data = nusc.get('sample_data', my_sample['data'][sensor])
cam_front_right_data
nusc.render_sample_data(cam_front_right_data['token'])
sensor = 'LIDAR_TOP'
lidar_top_data = nusc.get('sample_data', my_sample['data'][sensor])
lidar_top_data
nusc.render_sample_data(lidar_top_data['token'])
sensor = 'RADAR_BACK_LEFT'
radar_back_left_data = nusc.get('sample_data', my_sample['data'][sensor])
radar_back_left_data
nusc.render_sample_data(radar_back_left_data['token'])
sensor = 'RADAR_BACK_RIGHT'
radar_back_right_data = nusc.get('sample_data', my_sample['data'][sensor])
radar_back_right_data
nusc.render_sample_data(radar_back_right_data['token'])
sensor = 'RADAR_FRONT'
radar_front_data = nusc.get('sample_data', my_sample['data'][sensor])
radar_front_data
nusc.render_sample_data(radar_front_data['token'])
sensor = 'RADAR_FRONT_LEFT'
radar_front_left_data = nusc.get('sample_data', my_sample['data'][sensor])
radar_front_left_data
nusc.render_sample_data(radar_front_left_data['token'])
sensor = 'RADAR_FRONT_RIGHT'
radar_front_right_data = nusc.get('sample_data', my_sample['data'][sensor])
radar_front_right_data
nusc.render_sample_data(radar_front_right_data['token'])


##............................................................Q.4.........................................................
# 4.(1)
import cv2

# Load an color image
img = cv2.imread('C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes/image.jpg')

# Show image
cv2.imshow('C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes/image.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#.........................................................................................................................
#4.(2a)
import open3d as o3d
import numpy as np

seg_name="C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes/lidarseg/v1.0-mini/3fae29536ac04b02a45fe07a4a994bdf_lidarseg.bin"
seg=np.fromfile(seg_name, dtype=np.uint8)

color = np.zeros([len(seg), 3])
color[:, 0] = seg/32
color[:, 1] = seg/32
color[:, 2] = seg/32

sege ="C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151610946785.pcd.bin"
scan = np.fromfile(sege, dtype=np.float32)
points = scan.reshape((-1, 5))[:, :4]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])

#.........................................................................................................................
# 4.(2b)
import numpy as np
import open3d as o3d
import cv2, random,os
#Define PCD Path 
input_path= r"C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes/samples/LIDAR_TOP"

#Picking Random file
dataname = random.choice([
    x for x in os.listdir(input_path)
    if os.path.isfile(os.path.join(input_path, x))
])
Filename_rand= (os.path.join(input_path, dataname)) 
print('Displayed image is ',Filename_rand)

#Converting PCD to points
pcd_data=Filename_rand
data=np.fromfile(pcd_data, dtype=np.float32)
pointsdata = data.reshape((-1, 5))[:, :4]
print(pointsdata)

#Value of Z
Z=pointsdata[:,2]
print('The value of Z is ',Z)

#Value of Intensity
intensity=pointsdata[:,3]
print('The value of intensity is ',intensity)

#For height
color = np.zeros([len(Z), 3])
color[:, 0] = Z[1]/10
color[:, 1] = 0.5
color[:, 2] = 0.5


pcdimage = o3d.geometry.PointCloud()
pcdimage.points = o3d.utility.Vector3dVector(pointsdata[:, :3])
pcdimage.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcdimage])

#For Intensity
color = np.zeros([len(intensity), 3])
color[:, 0] = intensity[1]/10
color[:, 1] = 0.5
color[:, 2] = 0.5


pcdimage = o3d.geometry.PointCloud()
pcdimage.points = o3d.utility.Vector3dVector(pointsdata[:, :3])
pcdimage.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcdimage])

#Visualize by Segement
sem=np.fromfile(Filename_rand, dtype=np.uint8)


color = np.zeros([len(sem), 3])
color[:, 0] = sem/20
color[:, 1] = 0.5
color[:, 2] = 0.5
pcdimage = o3d.geometry.PointCloud()
pcdimage.points = o3d.utility.Vector3dVector(pointsdata[:, :3])
pcdimage.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcdimage])   

#................................................................................................................................

# 4.(3d(i))
import os.path as osp
import random
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

# 4.(3C)............................................

p = RadarPointCloud.from_file("C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes/samples/RADAR_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__RADAR_FRONT_RIGHT__1533151604573237.pcd")
p = p.points.T
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(p[:, :3])
pcl.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcl])

# 4.(3d(i))..........................................
nusc = NuScenes(version='v1.0-mini', dataroot='C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes', verbose=True)
random_number=random.randint(0,100)
my_sample = nusc.sample[random_number]  

pointsensor_token = my_sample['data']['RADAR_FRONT']
pointsensor = nusc.get('sample_data', pointsensor_token)
pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
#Radar plot 

pc = RadarPointCloud.from_file(pcl_path)
data=np.asarray(pc.points.astype(dtype=np.float32()))

pointsdata = data.reshape((-1, 18))[:, :17]
print(pointsdata)

#Height
Z=data[2]


#Velocity
vx=data[6]
vy=data[7]

#For height
color = np.zeros([len(Z), 3])
color[:, 0] = Z/10
color[:, 1] = 0.5
color[:, 2] = 0.5


pcdimage = o3d.geometry.PointCloud()
pcdimage.points = o3d.utility.Vector3dVector(pointsdata[:, :3])
pcdimage.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcdimage])
#...................................................................................................................................

# 4.(3d(ii))
import os.path as osp
import random
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
#from data_classes import RadarPointCloud
from nuscenes.utils.data_classes import RadarPointCloud

nusc = NuScenes(version='v1.0-mini', dataroot='C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes', verbose=True)
random_number=random.randint(0,100)
my_sample = nusc.sample[random_number]  

pointsensor_token = my_sample['data']['RADAR_FRONT']
pointsensor = nusc.get('sample_data', pointsensor_token)
pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
#Radar plot 

pc = RadarPointCloud.from_file(pcl_path)
#data=np.asarray(pc.points.astype(dtype=np.float32()))

data = pc.points.T

pointsdata = data.reshape((-1, 18))[:, :17]
print(pointsdata)

#Height
#Z=data[2]


#Velocity
vx=data[:,8]
vy=data[:,9]

#For height
color = np.zeros([len(vx), 3])
color[:, 0] = vx/10
color[:, 1] = vy/10
color[:, 2] = 0.5

pcdimage = o3d.geometry.PointCloud()
pcdimage.points = o3d.utility.Vector3dVector(pointsdata[:, :3])
pcdimage.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcdimage])
print('done')





# #............................................................Q.5.........................................................

# #5(1a.)
# #5(1a).
# ## To print calibration info (between Radar and Camera sensors) by referring code:
# # Step-1: Transportation of data points from Radar Sensor to the ego vehicle frame
sensor = 'RADAR_BACK_LEFT'
t_r_s = nusc.get('sample_data', my_sample['data'][sensor])
c_r_s = nusc.get('calibrated_sensor', t_r_s["calibrated_sensor_token"])
print("RADAR_BACK_LEFT translation_values is: ",c_r_s["translation"])
print("RADAR_BACK_LEFT rotation_values is: ",c_r_s["rotation"],"\n")
# Step-2: Transfomation of data points from ego vehicle frame to global frame
t_r_s = nusc.get('sample_data', my_sample['data'][sensor])
e_r_s = nusc.get('ego_pose',t_r_s["ego_pose_token"])
print("e_translation_values is: ",e_r_s["translation"])
print("e_rotation_values is: ",e_r_s["rotation"],"\n")
# Step-3: Transformation of data points from global frame to ego vehicle frame
sensor = 'CAM_BACK_LEFT'
t_c_s = nusc.get('sample_data', my_sample['data'][sensor])
c_c_s = nusc.get('calibrated_sensor', t_c_s["calibrated_sensor_token"])
print("CAM_BACK_LEFT translation_values is: ",c_c_s["translation"])
print("CAM_BACK_LEFT rotation_values is: ",c_c_s["rotation"],"\n")
# Step-4: Transformation of data points from ego vehicle frame to camera
t_s = nusc.get('sample_data', my_sample['data'][sensor])
e_c_s = nusc.get('ego_pose',t_c_s["ego_pose_token"])
print("e_translation_values is: ",e_c_s["translation"])
print("e_rotation_values is: ",e_c_s["rotation"],"\n")
# Step-5:  Finally, the radar data points are present in camera.


#5(1C)............................................................................................................................
import random
from nuscenes.nuscenes import NuScenes
from nusc_file import NuScenesExplorer
import random
from nuscenes.nuscenes import NuScenes
# from Nusc_file import NuScenesExplorer


nusc = NuScenes(version='v1.0-mini', dataroot='C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes', verbose=True)
random_number=random.randint(0,100)
my_sample = nusc.sample[random_number]  


#Radar plot 
nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='RADAR_FRONT')
NuScenesExplorer(nusc).map_pointcloud_to_image(my_sample['data']['RADAR_FRONT'],my_sample['data']['CAM_FRONT'])



#5(2d)............................................................................................................................
## To print calibration info (between Radar and Camera sensors) by referring code:
# Step-1: Transportation of data points from  Radar Sensor to the ego vehicle frame
sensor = 'LIDAR_TOP'
t_l_s = nusc.get('sample_data', my_sample['data'][sensor])
c_l_s = nusc.get('calibrated_sensor', t_l_s["calibrated_sensor_token"])
print("LIDAR_TOP translation_values is: ",c_l_s["translation"])
print("LIDAR_TOP rotation_values is: ",c_l_s["rotation"],"\n")
# Step-2: Transfomation ofdata points frpom ego vehicle frame to global frame
t_l_s = nusc.get('sample_data', my_sample['data'][sensor])
e_l_s = nusc.get('ego_pose',t_l_s["ego_pose_token"])
print("e_translation_values is: ",e_l_s["translation"])
print("e_rotation_values is: ",e_l_s["rotation"],"\n")
# Step-3: Transformation of data points from global frame to ego vehicle frame
sensor = 'CAM_BACK_LEFT'
t_c_s = nusc.get('sample_data', my_sample['data'][sensor])
c_c_s = nusc.get('calibrated_sensor', t_c_s["calibrated_sensor_token"])
print("CAM_BACK_LEFT translation_values is: ",c_c_s["translation"])
print("CAM_BACK_LEFT rotation_values is: ",c_c_s["rotation"],"\n")
# Step-4: Transformation ofdata points from ego vehicle frame to camera
t_s = nusc.get('sample_data', my_sample['data'][sensor])
e_c_s = nusc.get('ego_pose',t_c_s["ego_pose_token"])
print("e_translation_values is: ",e_c_s["translation"])
print("e_rotation_values is: ",e_c_s["rotation"],"\n")
# Step-5:  Finally, the radar data points is present in camera.

#............................................................................................................................
#5(2e)
nusc = NuScenes(version='v1.0-mini', dataroot='C:/Users/siddh/OneDrive/Pictures/Screenshots/Documents/NuScenes', verbose=True)
random_number=random.randint(0,100)
my_sample = nusc.sample[random_number]  

#lidar plot
nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', render_intensity=False)
NuScenesExplorer(nusc).map_pointcloud_to_image(my_sample['data']['LIDAR_TOP'],my_sample['data']['CAM_FRONT'])

#################################################################################################################################



