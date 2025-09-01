#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:50:51 2024

@author: leicheng
"""

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

SHOW = False

def save_data_dict(data_dict, filename, directory):
    """
    Save the data_dict as a pickle file to the specified directory with the filename.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Create the full path for the pickle file
    file_path = os.path.join(directory, filename)

    # Save the data_dict to a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)
        
        
def plot_point_cloud_3d(point_cloud):
    """
    Plots a 3D scatter plot of the point cloud data.

    Parameters:
    point_cloud (numpy array): A NumPy array of shape (N, 3) where N is the number of points,
                               and the columns represent x, y, z coordinates.
    """
    # Extract x, y, and z coordinates
    x_coords = point_cloud[:, 0]
    y_coords = point_cloud[:, 1]
    z_coords = point_cloud[:, 2]
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, s=1, c='blue', marker='o')  # s is the marker size, c is the color
    # # Add labels and title
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # ax.set_zlabel('Z Coordinate')
    # ax.set_title('3D Point Cloud Scatter Plot')
    # Show the plot
    plt.show()
    
     
def save_pkl_data(labeled_data, camera_data, lidar_data, label, cam_key, 
                   idx, idx2, filename, directory): 
    lid_key = labeled_data[cam_key][idx2][1]
    cam_box = labeled_data[cam_key][idx][2]
    lid_box = labeled_data[cam_key][idx2][3] #['bbox'] ['object_id']
    
    image = camera_data[cam_key]['image_pixels']
    if SHOW:
        # Display the image
        plt.imshow(image)
        plt.axis('off')  # Optional: Turn off axis labels
        plt.show()
    box = cam_box['bbox']
    cam_cls = cam_box['class'] # if label = 0
    cropped_image = image.crop((box['x1'], box['y1'], box['x2'], box['y2']))
    if SHOW:
        # Display the image
        plt.imshow(cropped_image)
        plt.axis('off')  # Optional: Turn off axis labels
        plt.show()
    center_point = [(box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2]
    #center_point = list(map(int, center_point))
    
    
    point_cloud = np.array(lidar_data[lid_key]['points'])[:, :3]  
    if SHOW:
        # Create a scatter plot of the x and y coordinates
        plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=1, c='blue')  # s is the marker size, c is the color
        #plot_point_cloud_3d(point_cloud)
        plt.show()  
    lid_cls = labeled_data[cam_key][idx2][2]['class']   
    box = lid_box['bbox']
    # Create a mask for points within the bounding box
    mask = (
        (point_cloud[:, 0] >= box['x_min']) & (point_cloud[:, 0] <= box['x_max']) &
        (point_cloud[:, 1] >= box['y_min']) & (point_cloud[:, 1] <= box['y_max']) &
        (point_cloud[:, 2] >= box['z_min']) & (point_cloud[:, 2] <= box['z_max'])
    )
    
    # Extract the points within the bounding box
    cropped_point_cloud = point_cloud[mask]
    if SHOW:
        # Create a scatter plot of the x and y coordinates
        plt.scatter(cropped_point_cloud[:, 0], cropped_point_cloud[:, 1], s=1, c='blue')  # s is the marker size, c is the color
        plt.show()
    lid_center_point = [(box['x_min'] + box['x_max']) / 2, (box['y_min'] + box['y_max']) / 2, (box['z_min'] + box['z_max']) / 2]
    
    # Create a dictionary to store the data
    #label = 1  # 1--same; 0--differ
    data_dict = {  ####lidar class, camera class
        'image': cropped_image,
        'img_cent': center_point,
        'lidar': cropped_point_cloud,
        'lid_cent': lid_center_point,
        'label': label,
        'cam_cls': cam_cls,
        'lid_cls': lid_cls,
        'cam_key': cam_key,
        'lid_key': lid_key,
        'cam_box': cam_box['bbox'],
        'lid_box': lid_box['bbox']             
        }
    # Create the filename by concatenating cam_key and idx with an underscore
    #filename = f"{cam_key}_{idx}.pkl"
    save_data_dict(data_dict, filename, save_dir)
    print(f"Save {filename}")    
    
    
def gen_train_data(labeled_data, camera_data, lidar_data, save_dir, n = 1, m = 3):     
    keys = sorted(list(labeled_data.keys()))
    for cam_key in keys:
        # Generate the index list
        length = len(labeled_data[cam_key])
        #n = 1  # how many numbers to exclude
        index_list = list(range(length))
        for idx in index_list:
            ######################### To save same object  #################################
            label = 1  # 1--same; 0--differ
            filename = f"{cam_key}_{idx}_{idx}.pkl"
            save_pkl_data(labeled_data, camera_data, lidar_data, label, cam_key, 
                               idx, idx, filename, save_dir)
            
            ######################### To save differ object  #################################
            label = 0  # 1--same; 0--differ
            # Create a copy of index_list and exclude the specific index (e.g., idx=1)
            index_list_copy = index_list.copy()
            index_list_copy.remove(idx)
            # Calculate the number of elements to select, [0, m]
            num_elements_to_select = min(max(length - n, 0), m)           
            # Randomly select num_elements_to_select elements from the copied list
            random_selection = random.sample(index_list_copy, num_elements_to_select)
            for idx2 in random_selection:
                filename = f"{cam_key}_{idx}_{idx2}.pkl"
                save_pkl_data(labeled_data, camera_data, lidar_data, label, cam_key, 
                                   idx, idx2, filename, save_dir)    
    

    
def gen_frame_data(data, save_dir): 
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)
    ######################### Gen Frames  #################################
    keys = sorted(list(data.keys()))
    for key in keys:
        filename = f"{key}.pkl"
        data_dict = data[key]
        save_data_dict(data_dict, filename, save_dir)
        print(f"Save {filename}")

            
            
  
    
    
if __name__ == '__main__':    
    # Define the base directory containing the Labeled_Data
    root     = r'/xdisk/caos/leicheng/FHWA'
    base_dir = r'/xdisk/caos/leicheng/FHWA/new_dataset'
    save_dir = r'/xdisk/caos/leicheng/FHWA/Lei_new_dataset' 
    
    # Iterate over each subdirectory
    for folder_name in sorted(os.listdir(base_dir)):
        print('folder_name: ', folder_name)
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # Iterate over each .pkl file in the subdirectory
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(folder_path, file_name)
                    # Assign data to variables based on specific strings in the file name
                    if 'Camera' in file_name:
                        with open(file_path, 'rb') as file:
                            camera_data = pickle.load(file)
                    elif 'Lidar' in file_name:
                        with open(file_path, 'rb') as file:
                            lidar_data = pickle.load(file)
                    elif 'label' in file_name:
                        with open(file_path, 'rb') as file:
                            labeled_data = pickle.load(file)
                            # Initialize an empty dictionary to store grouped data
                            grouped_data = defaultdict(list)
                            # Iterate through the labeled_data array
                            for item in labeled_data:
                                key = item[0]  # Extract the first element to be used as the key
                                grouped_data[key].append(item)  # Append the entire tuple to the corresponding key
                            # Convert defaultdict to a regular dictionary
                            labeled_data = dict(grouped_data)
            ####################  Process to generate Training data  ##############                
            gen_train_data(labeled_data, camera_data, lidar_data, save_dir)
            ####################  Process to generate frame data  ##############  
            # camera              
            gen_frame_data(camera_data, os.path.join(root,'lei_camera_newdataset'))
            # lidar              
            gen_frame_data(lidar_data,  os.path.join(root,'lei_lidar_newdataset'))

    
    
    # # Camera
    # keys = list(camera_data.keys())
    # cam_dets = camera_data[keys[0]]['detections'] #['box']
    # image = camera_data[keys[0]]['image_pixels']
    # # Display the image
    # plt.imshow(image)
    # plt.axis('off')  # Optional: Turn off axis labels
    # plt.show()
    
    # # Lidar
    # keys = list(lidar_data.keys())
    # lid_dets = lidar_data[keys[0]]['bboxes'] #['bbox']
    # point_cloud = lidar_data[keys[0]]['points']
    
    
    # # labeled
    # idx = 0
    # cam_key = labeled_data[idx][0]
    # cam_box = labeled_data[idx][2]
    # lid_key = labeled_data[idx][1]
    # lid_box = labeled_data[idx][3] #['bbox'] ['object_id']
    # keys = list(labeled_data.keys())



    








