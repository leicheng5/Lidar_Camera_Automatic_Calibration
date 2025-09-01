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
        
        
def save_train_data(img_fr, img_cents_fr, cam_box_fr, lid_fr, lid_cents_fr, lid_box_fr, 
                    cam_key, lid_key, save_dir): 
    # Create a dictionary to store the data
    data_dict = { 
        'image': img_fr,
        'img_cent': img_cents_fr,
        'cam_box': cam_box_fr,
        'lidar': lid_fr,
        'lid_cent': lid_cents_fr,
        'lid_box': lid_box_fr,
        'cam_key': cam_key,
        'lid_key': lid_key             
        }
    # Create the filename 
    filename = f"{cam_key}.pkl"
    save_data_dict(data_dict, filename, save_dir)
    print(f"Save {filename}")     
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
    #------------------------------------------------------#
    save_dir   = r'/xdisk/caos/leicheng/FHWA/Lei_new_dataset_FineTune'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    output_path = './finetune_calib_results'
    if not os.path.exists(output_path):
        os.makedirs(output_path)   
    video_path = './video_results'
    if not os.path.exists(video_path):
        os.makedirs(video_path)     

 
    
    
    #############################  Load calibration matrix   #############################
    #homography_matrix = np.load(os.path.join(output_path, r'homography_matrix_lid2cam_best.npy'), allow_pickle=True)
    
    #############################  Main   #############################
    #############################  Load images  #############################
    cam_filepaths, _, cam_boxes, cam_timestps = Data_Prep(cam_frame_dir, mode = 'camera')
    
    #############################  Load lidars  #############################
    lid_filepaths, _, lid_boxes, lid_timestps = Data_Prep(lid_frame_dir, mode = 'lidar')
    
    #############################  Matching images and lidars  #######################
    ## Associating the lidar and image by timestamps
    idx_lid, _ = find_nearest_betw_arr(lid_timestps, cam_timestps)
    
    ###########################  Traverse for each Paired lidar and image  #############################        
    for i in tqdm(range(len(cam_timestps))): #
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
        
        
        ##################### Center Points in a frame  ###############################
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

        ### end for one img ###
        img_cents_fr   = np.array(img_cents)
        lid_cents_fr   = np.array(lid_cents)  #= np.array(lid_cents)[:,:2]
        img_fr = np.array(image)
        lid_fr = np.array(lidar)
        cam_key = os.path.basename(img_path).rsplit(".", 1)[0]
        lid_key = os.path.basename(lid_path).rsplit(".", 1)[0]

        ## Save one train data frame
        save_train_data(img_fr, img_cents_fr, cam_box_fr, lid_fr, lid_cents_fr, lid_box_fr, 
                            cam_key, lid_key, save_dir)
        
        gc.collect()
                    
    ####    END for all imgs   #############################################################            
                   

        

        

