#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""

############### Lei ###################################
import os, glob, shutil, sys
# sys.path.append("./") # add search path: sys.path.append("../../")
# sys.path.append("./ImageYolo/")
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, parent_dir)
######################################################
from collections import defaultdict
from tqdm import tqdm
import datetime
import gc
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn

## For dataset
from utils.dataset_generator import Dataset, dataset_collate, pad_and_stack_tensors
from utils.utils import resize_image, get_classes

from discriminator import Comm_Feat_Discriminator

import cv2
#from moviepy.editor import ImageSequenceClip

import math
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial import distance
################################################################################\
def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data 
  
def crop_lidar(point_cloud, box):
    # Create a mask for points within the bounding box
    mask = (
        (point_cloud[:, 0] >= box['x_min']) & (point_cloud[:, 0] <= box['x_max']) &
        (point_cloud[:, 1] >= box['y_min']) & (point_cloud[:, 1] <= box['y_max']) &
        (point_cloud[:, 2] >= box['z_min']) & (point_cloud[:, 2] <= box['z_max'])
    )
    
    # Extract the points within the bounding box
    cropped_point_cloud = point_cloud[mask]
    return cropped_point_cloud 
    
    
def find_nearest_betw_arr(known_array, match_array):
    '''
    Based on match_array, to find the value in an known_array which is closest to an element in match_array
    return match_value and inx in known_array, and indices size is len(match_array)
    Returns:
        indices (array): Indices of the nearest values in known_values.
        nearest_values (array): Nearest values in known_values corresponding to the match_array.
    '''
    # known_array=np.array([1, 9, 33, 26,  5 , 0, 18, 11])
    # match_array=np.array([-1, 0, 11, 15, 33, 35,10,31])
    # i.e.:For -1 in match_array, the nearest value in known_values is 0 at index 5. For 33, the nearest value in known_values is 33 at index 2.
    # indices = array([5, 5, 7, 5, 2, 2, 7, 4])
    # known_array[indices] = array([0, 0, 11, 5, 33, 33, 11, 5])
    
    known_array = np.array(known_array)
    match_array = np.array(match_array)
    # Sort the known_values array and get the sorted indices
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted] 
    # Find the insertion points in the sorted_known_array for each match_array value
    idx = np.searchsorted(known_array_sorted, match_array)
    idx1 = np.clip(idx, 0, len(known_array_sorted)-1)
    idx2 = np.clip(idx - 1, 0, len(known_array_sorted)-1)
    # Calculate the differences between the nearest values and the match_array values
    diff1 = known_array_sorted[idx1] - match_array
    diff2 = match_array - known_array_sorted[idx2]
    # Determine the indices of the nearest values based on the differences
    indices = index_sorted[np.where(diff1 <= diff2, idx1, idx2)]
    return indices,known_array[indices]

def split_data(file_path, N=80, M=20, X=90, Y=10):
    """
    Splits data from the Train data TXT into training, validation, and test datasets.
    
    Parameters:
    - file_path (str): Path to the file containing the data paths.
    - N (float): Percentage of data to allocate to the training + validation set.
    - M (float): Percentage of data to allocate to the test set.
    - X (float): Percentage of the training + validation set to allocate to the training set.
    - Y (float): Percentage of the training + validation set to allocate to the validation set.
    """

    # Read the file and store the data paths in a list
    with open(file_path, 'r') as file:
        data_paths = file.read().splitlines()        
    # Calculate the number of total data points
    total_data = len(data_paths)  
    # # Shuffle the data to ensure random splitting
    # random.shuffle(data_paths)
    # Calculate the number of trainval data
    num_trainval = int(N * total_data / 100)    
    # Split into test and trainval (training + validation)
    trainval_set = data_paths[:num_trainval]
    test_set = data_paths[num_trainval:]
        
    # Further split the remaining set into training and validation
    num_train = int(X * len(trainval_set) / 100)
    train_set = trainval_set[:num_train]
    val_set = trainval_set[num_train:]   
    return train_set, val_set, test_set   



############  Data Loader  #####################################

def Data_Prep(data_dir = r'/xdisk/caos/leicheng/FHWA/lei_camera', mode = 'camera'): 
    filepaths = []
    frames    = []
    boxes     = []
    timestps  = []
    filter_cls_list = [0, 2, 7] #['person', 'car', 'truck']
    if mode == 'camera':
        # Iterate over each .pkl file in the subdirectory
        for file_name in sorted(os.listdir(data_dir)):           
            if file_name.endswith('.pkl'):
                file_path = os.path.join(data_dir, file_name)
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                filepaths.append(file_path)
                #frames.append(data['image_pixels'])                
                timestp = float(file_name.replace(".pkl", ""))
                timestps.append(timestp)
                # Iterate over each bbox
                box_fr = []
                data_det = data['detections']

                for idx in range(len(data_det)):
                    if len(data_det[idx]) < 4:
                        continue
                    if data_det[idx]['class'] not in filter_cls_list:
                        continue
                    elif data_det[idx]['conf'] < 0.5:
                        continue
                    box_fr.append(data_det[idx]['bbox'])
    
                boxes.append(box_fr)
                    
    elif mode == 'lidar':
        # Iterate over each .pkl file in the subdirectory
        for file_name in sorted(os.listdir(data_dir)):           
            if file_name.endswith('.pkl'):      # and ('background' not in file_name)
                file_path = os.path.join(data_dir, file_name)
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                filepaths.append(file_path)
                #frames.append(data['points'])
                timestp = float(file_name.replace(".pkl", ""))
                timestps.append(timestp)   
                # Iterate over each bbox
                box_fr = []
                data_det = data['detections']
                for idx in range(len(data_det)):
                    box_fr.append(data_det[idx]['bbox'])
    
                boxes.append(box_fr)
                 
    return filepaths, frames, boxes, timestps   

       
                            
def Data_Generator(frame_idx, data_sequences):
    """
    Generate train data(Img and GT_BBox) with batch size
    """
    train_filename = data_sequences[frame_idx] 
    # ## load radar RAD
    with open(train_filename, 'rb') as file:
        data = pickle.load(file)
    
    ### Parse data ###
    image    = data['image']
    lidar    = data['lidar']
    img_cent = data['img_cent']
    lid_cent = data['lid_cent']
    label    = data['label']          #1--same
    img_cls  = data['cam_cls']
    lid_cls  = data['lid_cls']
    img_box  = data['cam_box']
    lid_box  = data['lid_box']

    return image, np.array(lidar), np.array(img_cent), np.array(lid_cent), label, img_cls, lid_cls


def resize_data(image, lidar, img_cent, lid_cent, input_shape, letterbox_image=False):
    image    = resize_image(image, input_shape, letterbox_image)  
    image_np = np.transpose(np.array(image), (2, 0, 1)) # pytorch-->[C, H, W]
    
    return image_np, np.array(lidar), np.array(img_cent), np.array(lid_cent)

# For DataLoader's collate_fn, used to organize batch data as tensor
def organize_batch_tensor(image, lidar, img_cent, lid_cent):
    images     = []
    lidars     = []
    img_cents  = []
    lid_cents  = []

    images.append(image)
    lidars.append(torch.from_numpy(lidar).type(torch.FloatTensor))
    img_cents.append(img_cent)
    lid_cents.append(lid_cent)


            
    images     = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    img_cents  = torch.from_numpy(np.array(img_cents)).type(torch.FloatTensor)
    lid_cents  = torch.from_numpy(np.array(lid_cents)).type(torch.FloatTensor)

    ### Pad lidar data ##
    lidars = pad_and_stack_tensors(lidars)

    return images, lidars, img_cents, lid_cents

def load_datapath(file_path):
    """
    Load data from the data path TXT.
    
    Parameters:
    - file_path (str): Path to the file containing the data paths.
    """

    # Read the file and store the data paths in a list
    with open(file_path, 'r') as file:
        data_paths = file.read().splitlines()        

    test_set = data_paths
        
    return test_set  

def plot_paired_points(img_fr, img_points, radar_points, i, suffix, path, matrix, mode='homography'):
    if len(matrix)==3: #mode=='homography':
        # Apply the perspective transformation using homography_matrix
        transformed_points = cv2.perspectiveTransform(radar_points.reshape(-1, 1, 2), matrix)
    elif len(matrix)==2:    
        # Apply the affine transformation using affine_matrix
        transformed_points = cv2.transform(radar_points.reshape(-1, 1, 2), matrix)
        
    # Create a new figure for each plot
    plt.figure(dpi=300)
    
    # Plot the image
    plt.imshow(img_fr)
    plt.axis('off')  # Turn off the axis

    # Plot the image points
    if img_points.any():
        plt.scatter(img_points[:, 0], img_points[:, 1], s=4, c='red', label='Image')

    # Get the image dimensions
    height, width = img_fr.shape[:2]
    
    # Filter out the points that are outside the image boundaries
    valid_points = []
    for pt in transformed_points:
        x, y = pt[0]
        if 0 <= x < width and 0 <= y < height:
            valid_points.append(pt)
    transformed_points = np.array(valid_points)
    
    # # Plot the radar points
    # plt.scatter(radar_points[:, 0], radar_points[:, 1], c='blue', label='Radar Points')
    
    # Plot the Projected Points
    if transformed_points.size>0:
        plt.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], s=4, c='green', label='Lidar')

    # Add legend
    plt.legend(loc='lower right', fontsize='small')

    # Save the plot as an image file with a filename based on the iteration index
    filename = f'paired_points_plot_{i}_{suffix}.png'
    plt.savefig(os.path.join(path,filename), dpi=300)

    # Show the plot
    #plt.show()

    # Clear the current figure and remove previously plotted points
    plt.clf()

    # Close the figure
    plt.close()
    
   
def create_video_from_images(path, output_filename):
    # Get the list of image files in the specified path
    image_files = glob.glob(os.path.join(path, '*.png') )
    
    # Sort the image files based on the index i in the filename
    image_files = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not image_files:
        print("No image files found. Exiting...")
        return

    # Read the first image to get the dimensions
    img = cv2.imread(image_files[0])
    height, width, _ = img.shape

    # Create a video writer object
    video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

    # Iterate over the image files and write them to the video
    for image_file in image_files:
        img = cv2.imread(image_file)
        video_writer.write(img)

    # Release the video writer
    video_writer.release()

    print(f"Video created successfully: {output_filename}")    
    
##############################################################################
def Get_reproject_err(lidar_points, img_points, homography_matrix, max_dis_cost=150):
    # Apply the perspective transformation using homography_matrix
    projected_lidar = cv2.perspectiveTransform(lidar_points.reshape(-1, 1, 2), homography_matrix)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = projected_lidar - img_points.reshape(-1, 1, 2)
    reprojection_errors = np.linalg.norm(reprojection_errors, axis=-1)
    reprojection_error = np.mean(reprojection_errors[reprojection_errors <= max_dis_cost])
    #reprojection_error = np.mean(reprojection_errors)
    # Output  mean reprojection error
    print("Reprojection Error:", reprojection_error)
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("RMSE:", rmse)
    
    return reprojection_error, rmse
    
    
def lid_cam_Homography(lidar_points, img_points,validation_lidar_points, validation_img_points, 
                       method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995):
    # Use cv2.findHomography() to calculate the perspective transformation matrix: cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    # # #RANSAC
    homography_matrix, inliers = cv2.findHomography(lidar_points, img_points, method=method, 
                                                    ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters, confidence=confidence)
    
    # #LMEDS
    # homography_matrix, inliers = cv2.findHomography(lidar_points, img_points, method=cv2.LMEDS
    #                                               , ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995)
    
    # #RHO
    # homography_matrix, inliers = cv2.findHomography(lidar_points, img_points, method=cv2.RHO
    #                                               , ransacReprojThreshold=10.0, maxIters=2000, confidence=0.995)
    
    # Output the homography matrix
    print("Homography Matrix:",homography_matrix)
    
    
    ## Outliers and Inliers
    # Apply the perspective transformation using homography_matrix
    transformed_points = cv2.perspectiveTransform(lidar_points.reshape(-1, 1, 2), homography_matrix)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points - img_points.reshape(-1, 1, 2)
    mean_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Mean Reprojection Error:", mean_reprojection_error)
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("RMSE:", rmse)
    
    
    ## Inliers
    inliers_points = lidar_points[inliers.ravel()==1] #inliers.ravel().astype(bool)
    # Get Projected Points corresponding to inliers
    transformed_inliers = transformed_points[inliers.ravel()==1]
    inliers_reprojection_errors = reprojection_errors[inliers.ravel()==1]
    mean_reprojection_error_inliers = np.mean(np.linalg.norm(inliers_reprojection_errors, axis=-1))
    # Print the inliers and their Mean Reprojection Error
    #print("Inliers:", inliers_points)
    print("Number of inliers:", len(inliers_points))
    print("Mean Reprojection Error (inliers):", mean_reprojection_error_inliers)
    # Calculate Inliers RMSE
    inliers_rmse = np.sqrt(np.mean(np.square(inliers_reprojection_errors)))
    print("Inliers RMSE:", inliers_rmse)
    
    ## Validation points
    # Apply the perspective transformation using homography_matrix
    transformed_points_val = cv2.perspectiveTransform(validation_lidar_points.reshape(-1, 1, 2), homography_matrix)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_val - validation_img_points.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Validation-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("Validation-RMSE:", val_rmse)
    
    # val_reprojection_error, val_rmse = Get_reproject_err(validation_lidar_points, validation_img_points, homography_matrix)
    
    return homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val



def Do_Calibration(data, val_data):
    # Construct camera image coordinates and lidar image coordinates
    img_points   = data[:,0]
    lidar_points = data[:,1]
    ####################################################################################
    # Use additional Validation points
    validation_img_points   = val_data[:,0]
    validation_lidar_points = val_data[:,1]
    # # Remove outliers from the data points
    # filtered_indices = remove_outliers_zscore(img_points, threshold=3)
    # img_points       = img_points[filtered_indices]
    # lidar_points     = lidar_points[filtered_indices]
    ####################################################################################
    
    USE_findHomography   = 1
    if USE_findHomography:
        ### For all points
        homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
            lid_cam_Homography(lidar_points, img_points, validation_lidar_points, validation_img_points, 
                              method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
    
    return homography_matrix  



 ####################################################################################
def select_points_with_grid(data, image_width = 1440, image_height = 1080, num_cols = 288, num_rows = 216, interval=1):
    # Image size and number of regions
    #num_cols = 360 #288 #256 #128#80#64#32#16   #1440
    #num_rows = 270 #216 #144 #72 #60#48#24#12   #1080
    
    data = np.array(data)
    pixel_coords = data[:,0]  # pixel_coord[0]-> width; pixel_coord[1]-> height
    lid_coords   = data[:,1]
    ####################################################################################
    # Calculate the region size
    region_width = image_width // num_cols
    region_height = image_height // num_rows
    print("Region Width:", region_width)
    print("Region Height:", region_height)

    # Calculate the coordinates of region centers
    region_centers = []
    for row in range(num_rows):
        for col in range(num_cols):
            center_x = col * region_width + region_width // 2
            center_y = row * region_height + region_height // 2
            region_centers.append((center_x, center_y))

    # Assign pixels to the respective regions they fall into
    assigned_regions = {}
    for pixel_coord, img_rad_coord in zip(pixel_coords, data):
        region_index = (pixel_coord[1] // region_height) * num_cols + (pixel_coord[0] // region_width)
        
        if region_index in assigned_regions:
            # If the closest region is already in the assigned_regions dictionary,
            # append the current pixel_coord to the list of assigned pixels for that region.
            assigned_regions[region_index].append(img_rad_coord)
        else:
            # If the closest region is not yet present in the assigned_regions dictionary,
            # create a new entry in the dictionary with the closest_region as the key
            # and a list containing only the current pixel_coord as the value.
            assigned_regions[region_index] = [img_rad_coord]
    
    
    # Retrieve pixels from each region in a specific pattern
    selected_pixels = []
    for row in range(0, num_rows, interval):  # Iterate over rows with a step of interval=2
        for col in range(0, num_cols, interval):  # Iterate over columns with a step of interval=2
            current_region = row * num_cols + col  # Calculate the current region index
            if current_region in assigned_regions:  # Check if the current region has assigned pixels
                region_pixels = assigned_regions[current_region]  #print(assigned_regions.keys())
                if region_pixels: # Check if the region has any pixels
                    #selected_pixel = random.choice(region_pixels)  # Select a random pixel from the region
                    # Calculate the distance between each pixel and the region center
                    distances = [math.sqrt((pixel[0][0] - region_centers[current_region][0])**2 +
                                           (pixel[0][1] - region_centers[current_region][1])**2)
                                 for pixel in region_pixels]
                    # Find the index of the pixel with the minimum distance
                    closest_pixel_index = distances.index(min(distances))
                    selected_pixel = region_pixels[closest_pixel_index]  # Select the closest pixel
                    #selected_pixel = region_pixels[0]  # Select the closest pixel
                    selected_pixels.append(selected_pixel)
                    
                    
    # Check the number of selected pixels
    print("Number of selected paired points:", len(selected_pixels))
    if len(selected_pixels) < 9:
        print("Insufficient pixels selected. Please collect more data.")
    
    # Convert the selected pixels to a numpy array
    selected_pixels_array = np.array(selected_pixels)
    
    return selected_pixels_array
    
##############################################################################
def construct_point_pairs(lidar_ctd, cam_ctd, homography_matrix, max_dis_cost=150):
    if lidar_ctd.size and cam_ctd.size: #lidar and camera's detections are all not empty
        ####### Apply the perspective transformation using homography_matrix
        projected_lidars = cv2.perspectiveTransform(lidar_ctd.reshape(-1, 1, 2), homography_matrix)
        
        pairs_ctd = lid2cam_Association(np.squeeze(projected_lidars, axis=1), lidar_ctd, cam_ctd, max_dis_cost)      
    return pairs_ctd    



def lid2cam_Association(projected_lidars, lidar_ctd, cam_ctd, max_dis_cost):
    # Match lidar to camera, set max_dis_cost per your need
    matched, unmatched_lidar, unmatched_camera = \
        match_cam_to_lid(cam_ctd, projected_lidars, max_dis_cost) 
   
              
    pairs_ctd =[]
    # Deal with matched
    if len(matched) > 0: 
        for cam_idx, lid_idx in matched:
            cam_xy = cam_ctd[cam_idx] 
            rad_xy = lidar_ctd[lid_idx] 
            pairs_ctd.append([cam_xy,rad_xy])

    
    return np.array(pairs_ctd)    

##############################################################################
def greedy_cost_matching(cost_matrix):
    '''
    # Example usage
    cost_matrix = np.array([[14.289646, 210.91754, 392.2918, 2484.9646, 1291.036, 1546.9495],
                            [353.48056, 129.88396, 60.556602, 2152.5747, 979.81274, 1248.577],
                            [385.11633, 160.10587, 22.73597, 2131.765, 936.5885, 1203.7891],
                            [929.2216, 1152.9083, 1333.636, 3378.1528, 2227.66, 2472.841],
                            [226.83803, 24.797318, 183.06181, 2274.8672, 1097.7715, 1362.101]])
    
    matches, row_ind, col_ind = greedy_cost_matching(cost_matrix)
    print("Matches (row, col):", matches)
            Matches (row, col): [(0, 0), (4, 1), (2, 2), (1, 3), (3, 4)]

    '''
    # Get the shape of the cost_matrix
    num_rows, num_cols = cost_matrix.shape

    # Initialize match list
    matches = []

    # Create flags for row and column usage, initialized to False
    row_used = [False] * num_rows
    col_used = [False] * num_cols

    # Continue matching until all possible matches are completed
    while len(matches) < min(num_rows, num_cols):
        # Find the minimum value in the unused rows and columns
        min_value = float('inf')
        min_row = -1
        min_col = -1
        for r in range(num_rows):
            if row_used[r]:
                continue
            for c in range(num_cols):
                if col_used[c]:
                    continue
                if cost_matrix[r, c] < min_value:
                    min_value = cost_matrix[r, c]
                    min_row = r
                    min_col = c

        # Mark the matched row and column
        matches.append((min_row, min_col))
        row_used[min_row] = True
        col_used[min_col] = True
        
    # Extract the first and second columns of matches
    row_ind, col_ind = zip(*matches)

    return matches, row_ind, col_ind



def match_cam_to_lid(trackers, detections, max_dis_cost=80):
    # Initialize 'cost_matrix'
    cost_matrix = np.zeros(shape=(len(trackers), len(detections)), dtype=np.float32)

    # Populate 'cost_matrix'
    for t, tracker in enumerate(trackers):
        for d, detection in enumerate(detections):
            cost_matrix[t,d] = distance.euclidean(tracker, detection)        

    # Produce matches by using the greedy algorithm to minimize the cost_distance
    #row_ind, col_ind = linear_assignment(cost_matrix) #`row_ind array` for `tracks`,`col_ind` for `detections`
    match_ind, row_ind, col_ind = greedy_cost_matching(cost_matrix)
    
    # Populate 'unmatched_trackers'--camera
    unmatched_trackers = []
    for t in np.arange(len(trackers)):
        if t not in row_ind: #`row_ind` for `tracks`
            unmatched_trackers.append(t)

    # Populate 'unmatched_detections'--lidar
    unmatched_detections = []
    for d in np.arange(len(detections)):
        if d not in col_ind:#`col_ind` for `detections`
            unmatched_detections.append(d)

    # Populate 'matches'
    matches = []
    for t_idx, d_idx in zip(row_ind, col_ind):
        # Check for cost distance threshold.
        # If cost is very high then unmatched (delete) the track
        if cost_matrix[t_idx,d_idx] < max_dis_cost:
            matches.append([t_idx, d_idx])
        else:
            unmatched_trackers.append(t_idx)
            unmatched_detections.append(d_idx)

    # Return matches, unmatched detection and unmatched trackers
    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)
#######
def filter_point_pairs(point_pairs, width=1920, height=1080):
    """
    Given an array of shape (N, 2, 2), where point_pairs[:,0] are image coordinates,
    return a new array containing only those pairs whose image coords lie within
    [0, width) × [0, height).

    Parameters:
        point_pairs: np.ndarray of shape (N, 2, 2)
        width:       image width (default 1920)
        height:      image height (default 1080)

    Returns:
        filtered:   np.ndarray of shape (M, 2, 2), M ≤ N
    """
    # Extract the image coordinates (first point of each pair)
    img_coords = point_pairs[:, 0, :]  # shape (N, 2)

    # Check bounds
    xs = img_coords[:, 0]
    ys = img_coords[:, 1]
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

    # Filter the pairs
    return point_pairs[inside]



    
###########################################################################
if __name__ == "__main__":
   
    #############################  Parameters   #############################
    torch.cuda.empty_cache()
    #---------------------------------#
    Cuda            = torch.cuda.is_available()
    #------------------------------------------------------#
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #---------------------------------------------------------------------#
    #data_dir        = r'/xdisk/caos/leicheng/FHWA/Labeled_Data'
    cam_frame_dir   = r'/xdisk/caos/leicheng/FHWA/lei_camera_dataset3'
    lid_frame_dir   = r'/xdisk/caos/leicheng/FHWA/lei_lidar_dataset3'
    classes_path    = '../coco_classes.txt' #os.path.join(parent_dir, 'coco_classes.txt')
    train_data_file = '/xdisk/caos/leicheng/FHWA/train_Dataset3_paths.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = '../best_model/best_epoch_weights.pth'
    #------------------------------------------------------#
    input_shape     = [32, 32]  #PyTorch--> [C, H, W]; CiFar dataset
    #------------------------------------------------------#
    max_dis_cost    =  150
    #------------------------------------------------------#
    output_path = './calib_results'
    if not os.path.exists(output_path):
        os.makedirs(output_path)   
    save_path = './video_results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)     

    # ###################    Dataset  ###################### 
    # #------------------------------------------------------#
    # #   Get classes
    # #------------------------------------------------------#
    # class_names, num_classes = get_classes(classes_path)
    # #---------------------------#
    # #   Split train and validation
    # #---------------------------#
    # #_, _, test_set = split_data(train_data_file, N=80, M=20, X=90, Y=10)
    # test_set     = load_datapath(train_data_file)
    # num_test    = len(test_set)  
    
    
    #############################  Load calibration matrix   #############################
    #homography_matrix_0 = np.load(os.path.join(output_path, r'homography_matrix_lid2cam.npy'), allow_pickle=True)
    homography_matrix_0 = np.load(os.path.join(output_path, r'Homography_matrix_lid2cam_0.npy'), allow_pickle=True)
    # homography_matrix_0 = np.load(os.path.join(output_path, r'new_H.npy'), allow_pickle=True)
    # homography_matrix_0        = homography_matrix_0.squeeze()

    homography_matrix_best = homography_matrix_0.copy()
    #############################  Main   #############################
    #############################  Load images  #############################
    cam_filepaths, _, cam_boxes, cam_timestps = Data_Prep(cam_frame_dir, mode = 'camera')
    
    #############################  Load lidars  #############################
    lid_filepaths, _, lid_boxes, lid_timestps = Data_Prep(lid_frame_dir, mode = 'lidar')
    
    #############################  Matching images and lidars  #######################
    ## Associating the lidar and image by timestamps
    idx_lid, _ = find_nearest_betw_arr(lid_timestps, cam_timestps)
    
    ###########################  Paired radar and image  #############################     
    # Get paired lidar and image based on common feats
    point_pairs = [] # List to store images and lidar 's centers
    #point_pairs_100 = []
    
    for i in tqdm(range(len(cam_timestps))):
    #for i in tqdm(range(101)):
        print('img idx : ', i)
        #### non empty check
        # camera
        img_path     = cam_filepaths[i]
        # Check if the file is non-empty
        if os.path.getsize(img_path) == 0:
            continue
        image        = read_pkl(img_path)
        image        = image['pixels']
        cam_box_fr   = cam_boxes[i]

        # lidar
        # Get radar index based on the img_idx
        lid_idx      = idx_lid[i]
        lid_path     = lid_filepaths[lid_idx]
        # Check if the file is non-empty
        if os.path.getsize(lid_path) == 0:
            continue
        lidar        = read_pkl(lid_path)
        lidar        = np.array(lidar['points'])[:, :3]  
        lid_box_fr   = lid_boxes[lid_idx]
        
        
        #####################  Points in a frame  ###############################
        img_cents = []
        # Iterate over each bbox    
        for j in range(len(cam_box_fr)):  # iterate over each img bbox
            # Get centers
            box = cam_box_fr[j]
            img_cent = [(box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2]
            # Append the img_cent to the img_cents list
            img_cents.append(img_cent)

        lid_cents = []    
        for k in range(len(lid_box_fr)):  # iterate over each lidar bbox             
            # Get centers
            box = lid_box_fr[k]
            lid_cent = [(box['x_min'] + box['x_max']) / 2, (box['y_min'] + box['y_max']) / 2, (box['z_min'] + box['z_max']) / 2]
            lid_cents.append(lid_cent)

        ### loop end: Got all cent points 
        img_points   = np.array(img_cents)
        lidar_points = np.array(lid_cents)[:,:2]
        
        # Apply the perspective transformation using homography_matrix
        #projected_points = cv2.perspectiveTransform(lidar_points.reshape(-1, 1, 2), homography_matrix_best)
        
        point_pairs_fr = construct_point_pairs(lidar_points, img_points, homography_matrix_best, 
                                               max_dis_cost)
        
        point_pairs.append(point_pairs_fr)         

        ####    Perform Calibration  every after accumulating 100 frames   #####################################################    
        Frames_GAP = 100
        if len(point_pairs) % Frames_GAP == 0  or len(point_pairs) == len(cam_timestps): #total_length=len(cam_timestps)
            point_pair_100 = np.concatenate(point_pairs, axis=0)
            point_pair_100 = filter_point_pairs(point_pair_100, width=1920, height=1080)
       
        
            
            point_pairs_grid = select_points_with_grid(point_pair_100, image_width = 1440, image_height = 1080, 
                                                       num_cols = 36, num_rows = 27, interval=1)
                
            homography_matrix = Do_Calibration(data=point_pairs_grid, val_data=point_pair_100)   
            
            ####    Get reprojection error for best and current calibration matrix
            img_coords = point_pair_100[:,0]
            lid_coords = point_pair_100[:,1]
            rep_err_now, rmse_now = Get_reproject_err(lid_coords, img_coords, homography_matrix, max_dis_cost)
            rep_err_best, rmse_best = Get_reproject_err(lid_coords, img_coords, homography_matrix_best, max_dis_cost)
            
            # Output  mean reprojection error
            print("Reprojection Error Now<->Best: ", rep_err_now, "<->",rep_err_best)
            # Project point pairs on image
            img_fr = np.array(image)
            img_coords_grid = point_pairs_grid[:,0]
            lid_coords_grid = point_pairs_grid[:,1]
            plot_paired_points(img_fr, img_coords_grid, lid_coords_grid, i, 'best', output_path, homography_matrix_best) 
            plot_paired_points(img_fr, img_coords, lid_coords, i+10000, 'best', output_path, homography_matrix_best) 
            np.save(os.path.join(output_path, f'Homography_matrix_lid2cam_best_{i}.npy'), homography_matrix_best)
            
            plot_paired_points(img_fr, img_coords_grid, lid_coords_grid, i, 'now', output_path, homography_matrix) 
            plot_paired_points(img_fr, img_coords, lid_coords, i+10000, 'now', output_path, homography_matrix) 
            np.save(os.path.join(output_path, f'Homography_matrix_lid2cam_now_{i}.npy'), homography_matrix)
                        
            # Plot calibration results
            #show_calib_result(lidar_points, img_points, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_lidar_points, transformed_points_val)
            
            # Re-assign the best matrix
            #homography_matrix_best = homography_matrix.copy()
            if rep_err_now < rep_err_best:
                print("Best is changed: ", rep_err_now, "<-",rep_err_best)
                homography_matrix_best = homography_matrix.copy()
            # Save  as an .npy file
            np.save(os.path.join(output_path, 'Homography_matrix_lid2cam_best.npy'), homography_matrix_best) 
            ####    Create video   #############################################################            
            # output_filename = 'LC_paired_points.avi'
            # create_video_from_images(save_path, output_filename)
            ###########################  PLot Finished  #############################                    

    
    
