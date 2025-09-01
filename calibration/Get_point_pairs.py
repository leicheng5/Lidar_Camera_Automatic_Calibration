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
    output_path = './calib_results'
    if not os.path.exists(output_path):
        os.makedirs(output_path)    

    ###################    Dataset  ###################### 
    #------------------------------------------------------#
    #   Get classes
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    #---------------------------#
    #   Split train and validation
    #---------------------------#
    #_, _, test_set = split_data(train_data_file, N=80, M=20, X=90, Y=10)
    test_set     = load_datapath(train_data_file)
    num_test    = len(test_set)  
    
    ###################    Load Model  ###################### 
    #------------------------------------------------------#
    #   Build Model
    #------------------------------------------------------#  
    '''Discriminator MODEL LOADING'''    
    model = Comm_Feat_Discriminator(num_class=num_classes, token_len = 256)    
    #------------------------------------------------------#
    #   Load weights
    #------------------------------------------------------#
    model.load_state_dict(torch.load(model_path, map_location=device))
    model    = model.eval()   #test
    print('{} model, and classes loaded.'.format(model_path))
    #------------------------------------------------------#
    #   Use CUDA
    #------------------------------------------------------#
    if Cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()   
        
    
    #############################  Load images  #############################
    cam_filepaths, _, cam_boxes, cam_timestps = Data_Prep(cam_frame_dir, mode = 'camera')
    
    #############################  Load lidars  #############################
    lid_filepaths, _, lid_boxes, lid_timestps = Data_Prep(lid_frame_dir, mode = 'lidar')
    
    #############################  Matching images and lidars  #######################
    ## Associating the lidar and image by timestamps
    idx_lid, _ = find_nearest_betw_arr(lid_timestps, cam_timestps)
    
    ###########################  Paired radar and image  #############################     
    # Get paired lidar and image based on common feats
    paired_ctds = [] # List to store images and lidar 's centers
    
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
        
        
        #####################  Avoid repeat compare  ###############################
        #uses the already_compared_same_j and already_compared_same_k variables to keep track 
        #of the indices that have already been compared and found to be the same. 
        #If either of these variables matches the current indices j and k, 
        #the loop continues to the next iteration without performing the comparison again.
        already_compared_same_j = None
        already_compared_same_k = None
        
        # Iterate over each bbox    
        for j in range(len(cam_box_fr)):  # iterate over each img bbox
            # crop data
            box = cam_box_fr[j]
            cropped_image = image.crop((box['x1'], box['y1'], box['x2'], box['y2']))
            # Get centers
            img_cent = [(box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2]

            
            for k in range(len(lid_box_fr)):  # iterate over each lidar bbox 
            
                # # Check if already compared and found to be the same, then skip this img iteration
                # if already_compared_same_j == j:
                #     continue
                # # Check if already compared and found to be the same, then skip this lid iteration
                # if already_compared_same_k == k:
                #     continue
                
                # crop data
                box = lid_box_fr[k]
                cropped_lidar = crop_lidar(lidar, box)
                # Get centers
                lid_cent = [(box['x_min'] + box['x_max']) / 2, (box['y_min'] + box['y_max']) / 2, (box['z_min'] + box['z_max']) / 2]

                #----------------------#
                #   Deep Learning Model --> Forward Propagation
                #----------------------#
                # construct batch data
                image1, lidar1, img_cent1, lid_cent1 = resize_data(cropped_image, cropped_lidar, 
                                                                   img_cent, lid_cent, input_shape, letterbox_image=False)
                images, lidars, img_cents, lid_cents = organize_batch_tensor(image1, lidar1, 
                                                                             img_cent1, lid_cent1)

                # prediction
                outputs, img_cls_res, lid_cls_res, _ = model(images, lidars, img_cents, lid_cents)
                
                # Calculate binary classification result
                pred_label = torch.sigmoid(outputs) > 0.5    
                print('same' if pred_label.int() == 1 else 'diff')
                # plt.imshow(images[0])
                # plt.show()

                
                if pred_label.int() == 1:
                    paired_ctds.append([img_cent, lid_cent, [torch.sigmoid(outputs)[0][0].item()]])
                    # Mark the current indices as already compared and found to be the same
                    already_compared_same_j = j
                    already_compared_same_k = k
        ### loop end            
        gc.collect()
                    
                    
    ####   END   #############################################################                
    # # Save paired_ctds as an .npy file
    # file_name = 'paired_ctds_2d.npy'
    # np.save(os.path.join(output_path, file_name), paired_ctds_2d, allow_pickle=True) 
    
    # Save paired_ctds as an .txt file
    file_name = 'paired_ctds.txt'
    # Save to a text file
    with open(os.path.join(output_path, file_name), 'w') as f:
        for img_cent, lid_cent, conf in paired_ctds:
            f.write(f"{img_cent} || {lid_cent} || {conf}\n")  
            
    # Save the list using pickle
    file_name = 'paired_ctds.pkl'
    with open(os.path.join(output_path, file_name), 'wb') as f:
        pickle.dump(paired_ctds, f) 
        
    # Extract the first two elements of the 3D coordinates
    paired_ctds_2d = [[img_cent, lid_cent[:2], conf] for img_cent, lid_cent, conf in paired_ctds]
    # Save the list using pickle
    file_name = 'paired_ctds_2d.pkl'
    with open(os.path.join(output_path, file_name), 'wb') as f:
        pickle.dump(paired_ctds_2d, f)               
        
        

        

