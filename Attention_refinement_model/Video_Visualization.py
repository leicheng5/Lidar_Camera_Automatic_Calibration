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
#############################  Lidar-Camera Matching    #######################################\
from scipy.spatial import distance    
def greedy_cost_matching(cost_matrix):
    """
    # Example usage
    cost_matrix = np.array([[14.289646, 210.91754, 392.2918, 2484.9646, 1291.036, 1546.9495],
                            [353.48056, 129.88396, 60.556602, 2152.5747, 979.81274, 1248.577],
                            [385.11633, 160.10587, 22.73597, 2131.765, 936.5885, 1203.7891],
                            [929.2216, 1152.9083, 1333.636, 3378.1528, 2227.66, 2472.841],
                            [226.83803, 24.797318, 183.06181, 2274.8672, 1097.7715, 1362.101]])
    
    matches, row_ind, col_ind = greedy_cost_matching(cost_matrix)
    print("Matches (row, col):", matches)
            Matches (row, col): [(0, 0), (4, 1), (2, 2), (1, 3), (3, 4)]

    """
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

def match_cam_to_lid(trackers, detections, max_dis_cost=100):
    """
    Match camera and LiDAR points based on a cost matrix and the greedy matching algorithm.
    
    Args:
        trackers (ndarray): Shape [N, 2], representing points from camera.
        detections (ndarray): Shape [M, 2], representing points from LiDAR.
        max_dis_cost (float): Maximum allowed distance for a valid match.
    
    Returns:
        matches (ndarray): Matched indices of shape [k, 2].
        unmatched_detections (ndarray): Indices of unmatched LiDAR points.
        unmatched_trackers (ndarray): Indices of unmatched camera points.
    """
    # Initialize 'cost_matrix'
    cost_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)

    # Populate 'cost_matrix' using Euclidean distances
    for t, tracker in enumerate(trackers):
        for d, detection in enumerate(detections):
            cost_matrix[t, d] = distance.euclidean(tracker, detection)

    # Produce matches using the greedy matching algorithm
    matches, row_ind, col_ind = greedy_cost_matching(cost_matrix)

    # Populate 'unmatched_trackers'
    unmatched_trackers = [t for t in range(len(trackers)) if t not in row_ind]

    # Populate 'unmatched_detections'
    unmatched_detections = [d for d in range(len(detections)) if d not in col_ind]

    # Finalize matches with threshold check
    filtered_matches = []
    for t_idx, d_idx in matches:
        if cost_matrix[t_idx, d_idx] < max_dis_cost:
            filtered_matches.append([t_idx, d_idx])
        else:
            unmatched_trackers.append(t_idx)
            unmatched_detections.append(d_idx)

    # Return results
    return np.array(filtered_matches), np.array(unmatched_detections), np.array(unmatched_trackers)

def lid_cam_Association(proj_lidar_ctd, lidar_ctd, cam_ctd, max_dis_cost):
    # Match lidar to camera, set max_dis_cost per your need
    matched, unmatched_lidar, unmatched_camera = \
        match_cam_to_lid(cam_ctd, proj_lidar_ctd, max_dis_cost) 
   
    cam_xy =[]
    lid_xy =[]
    lid_proj_xy =[]
    # Deal with matched
    if len(matched) > 0: 
        for cam_idx, lid_idx in matched:
            cam_xy.append(cam_ctd[cam_idx]) 
            lid_xy.append(lidar_ctd[lid_idx]) 
            lid_proj_xy.append(proj_lidar_ctd[lid_idx])
    
    return np.array(lid_proj_xy), np.array(lid_xy), np.array(cam_xy)      
    
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
    filter_cls_list = ['person', 'car', 'truck']
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
                    if data_det[idx]['name'] not in filter_cls_list:
                        continue
                    elif data_det[idx]['confidence'] < 0.5:
                        continue
                    box_fr.append(data_det[idx]['box'])
    
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
                data_det = data['bboxes']
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

def filter_points_within_img(lid_cent_proj, lid_cent, img_cent, width=1440, height=1080):
    """
    Filters out points in lid_cent_proj that are outside the image boundaries 
    and ensures lid_cent and img_cent are filtered consistently.

    Parameters:
        lid_cent (numpy.ndarray): Original lidar points, shape (N, 2).
        img_cent (numpy.ndarray): Corresponding image center points, shape (N, 2).
        lid_cent_proj (numpy.ndarray): Projected lidar points, shape (N, 2).
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        numpy.ndarray: Filtered lid_cent.
        numpy.ndarray: Filtered img_cent.
        numpy.ndarray: Filtered lid_cent_proj.
    """
    # Validate input dimensions
    assert lid_cent.shape == img_cent.shape == lid_cent_proj.shape, \
        "lid_cent, img_cent, and lid_cent_proj must have the same dimensions"
    assert lid_cent.shape[1] == 2, "Input arrays must have shape (N, 2)"
    
    # Boolean mask to identify points within boundaries
    valid_mask = (lid_cent_proj[:, 0] >= 0) & (lid_cent_proj[:, 0] < (width)) & \
                 (lid_cent_proj[:, 1] >= 0) & (lid_cent_proj[:, 1] < (height))
    # valid_mask = (lid_cent_proj[:, 0] >= 0) & (lid_cent_proj[:, 0] < width) & \
    #              (lid_cent_proj[:, 1] >= 0) & (lid_cent_proj[:, 1] < height)                     
    
    # Filter all points using the mask
    filtered_lid_cent = lid_cent[valid_mask]
    filtered_img_cent = img_cent[valid_mask]
    filtered_lid_cent_proj = lid_cent_proj[valid_mask]

    return filtered_lid_cent_proj, filtered_lid_cent, filtered_img_cent

def Compute_reproject_err(lidar_points, img_points, homography_matrix, width=1440, height=1080, max_dis_cost=150):
    # Apply the perspective transformation using homography_matrix
    lid_cent_proj = cv2.perspectiveTransform(lidar_points.reshape(-1, 1, 2), homography_matrix)
    
    lid_cent_proj, lid_cent, img_cent = lid_cam_Association(lid_cent_proj.squeeze(1), lidar_points, img_points, max_dis_cost)
    lid_cent_proj, lid_cent, img_cent = filter_points_within_img(lid_cent_proj, lid_cent, img_cent, width, height)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = lid_cent_proj - img_cent
    reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Reprojection Error:", reprojection_error)
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("RMSE:", rmse)
    
    return reprojection_error, rmse, lid_cent, img_cent

def Get_reproject_err(lidar_points, img_points, homography_matrix):
    # Apply the perspective transformation using homography_matrix
    lid_cent_proj = cv2.perspectiveTransform(lidar_points.reshape(-1, 1, 2), homography_matrix)
    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = lid_cent_proj - img_points.reshape(-1, 1, 2)
    reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Reprojection Error:", reprojection_error)
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("RMSE:", rmse)
    
    return reprojection_error, rmse

def plot_paired_points(img_fr, img_points, radar_points, i, path, matrix, mode='homography'):
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
    filename = f'paired_points_plot_{i}.png'
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
    cam_frame_dir   = r'/xdisk/caos/leicheng/FHWA/lei_camera_newdataset'
    lid_frame_dir   = r'/xdisk/caos/leicheng/FHWA/lei_lidar_newdataset'    
    classes_path    = '../coco_classes.txt' #os.path.join(parent_dir, 'coco_classes.txt')
    train_data_file = '/xdisk/caos/leicheng/FHWA/train_newDataset_paths.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = '../best_model/best_epoch_weights.pth'
    #------------------------------------------------------#
    input_shape     = [32, 32]  #PyTorch--> [C, H, W]; CiFar dataset
    #------------------------------------------------------#
    output_path = './DL_calib_results'
    if not os.path.exists(output_path):
        os.makedirs(output_path)   
    save_path = './DL_video_results'
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
    #homography_matrix = np.load(os.path.join(output_path, r'homography_matrix_lid2cam.npy'), allow_pickle=True)
    homography_matrix = np.load(r'./homography_matrix_lid2cam_best_init.npy', allow_pickle=True)
    new_matrix        = np.load(r'./test_FineTune/new_H.npy', allow_pickle=True)
    new_matrix        = new_matrix.squeeze()
    #############################  Main   #############################
    #############################  Load images  #############################
    cam_filepaths, _, cam_boxes, cam_timestps = Data_Prep(cam_frame_dir, mode = 'camera')
    
    #############################  Load lidars  #############################
    lid_filepaths, _, lid_boxes, lid_timestps = Data_Prep(lid_frame_dir, mode = 'lidar')
    
    #############################  Matching images and lidars  #######################
    ## Associating the lidar and image by timestamps
    idx_lid, _ = find_nearest_betw_arr(lid_timestps, cam_timestps)
    
    ###########################  Paired lidar and image  #############################     
    # Get paired lidar and image based on common feats
    paired_ctds = [] # List to store images and lidar 's centers
    improved_idx = [] 
    rep_err, rmse = 0, 0
    old_rep_err, old_rmse = 0, 0
    count = 0
    err_diff = 0
    
    for i in tqdm(range(len(cam_timestps))):
    #for i in tqdm(range(1)):
        print('img idx : ', i)
        #### non empty check
        # camera
        img_path     = cam_filepaths[i]
        # Check if the file is non-empty
        if os.path.getsize(img_path) == 0:
            continue
        image        = read_pkl(img_path)
        image        = image['image_pixels']
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

        ### loop end 
        img_points   = np.array(img_cents)
        lidar_points = np.array(lid_cents)[:,:2]
        img_fr = np.array(image)
        rep_err_org, rmse_org, new_lid_points, new_img_points = Compute_reproject_err(lidar_points, img_points, homography_matrix, width=img_fr.shape[1], height=img_fr.shape[0], max_dis_cost=150)
        rep_err_new, rmse_new = Get_reproject_err(new_lid_points, new_img_points, new_matrix)
        old_rep_err += rep_err_org
        old_rmse    += rmse_org
        if rep_err_org > rep_err_new:
            rep_err += rep_err_new
            rmse    += rmse_new
            matrix = new_matrix
            count += 1
            improved_idx.append(i)
            err_diff +=(rep_err_org - rep_err_new)
            print(f"{i}-th frame is improved")
        else:
            rep_err += rep_err_org
            rmse    += rmse_org
            matrix = homography_matrix
        plot_paired_points(img_fr, img_points, lidar_points, i, save_path, matrix)
        
        num_fr = i+1
        
        gc.collect()
    
    save_txt = './Final_Reprojection_Error.txt'
    with open(save_txt, 'w') as file:
        file.write(f"rep_err: {rep_err:.5f} \n")
        file.write(f"rmse: {rmse:.5f} \n")
        file.write(f"old_rep_err: {old_rep_err:.5f} \n")
        file.write(f"old_rmse: {old_rmse:.5f} \n")
        file.write(f"err_diff: {err_diff:.5f} \n")
        file.write(f"Total {num_fr} frames\n")
        file.write(f"{count} frames have improved \n")
        file.write(f"improved frames IDX : {improved_idx}\n")          
    ####    Create video   #############################################################            
    output_filename = 'LC_paired_points.avi'
    create_video_from_images(save_path, output_filename)
    # output_filename = 'LC_paired_points.mp4'
    # create_HQ_video_from_images(save_path, output_filename)
    ###########################  PLot Finished  #############################                    

        

        

