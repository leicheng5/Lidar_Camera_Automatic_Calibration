"""
Author: Lei Cheng
"""
import datetime
import os
import matplotlib.pyplot as plt
from functools import partial

import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
################################################################################
import cv2
from scipy.spatial import distance
import torch.nn.functional as F
from utils_finetune.utils import (ModelEMA, get_classes, seed_everything,
                         show_config, worker_init_fn, LossHistory)
from utils_finetune.utils_fit import fit_one_epoch
## For dataset
from utils_finetune.dataset_generator import Dataset, dataset_collate

from fine_tune_model import FineTune_Model
################################################################################
TEST = False  # False to generate calib_matrix
Intrinsic_Learning = True
  
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

    
def lid2cam_proj(lid_points, homography_matrix):
    """
    Apply the homography perspective transformation to LiDAR points to project them into the camera frame.
    
    Args:
        lid_points (Tensor or ndarray): Shape [B, N, 2], where B is the batch size and N is the number of points.
        homography_matrix (ndarray): Shape [3, 3], the homography matrix.
    
    Returns:
        projected_lid_points (ndarray): Shape [B, N, 2], transformed points in the camera frame.
    """
    # For batch processing
    batch_size = len(lid_points)
    projected_lid_points = []

    for b in range(batch_size):
        # Apply the homography transformation for each batch item
        batch_projected = cv2.perspectiveTransform(lid_points[b].reshape(-1, 1, 2).cpu().numpy(), homography_matrix[b].cpu().detach().numpy())
        projected_lid_points.append(torch.tensor(batch_projected.squeeze(1), dtype=torch.float32, device=lid_points[b].device))
        
    return projected_lid_points  

def torch_lid2cam_proj(lid_points, homography_matrix):
    """
    Apply the homography perspective transformation using PyTorch.

    Args:
        lid_points (list of Tensor): List of 2D tensors, each of shape [N, 2], where N is the number of points.
        homography_matrix (Tensor): Tensor of shape [B, 3, 3], homography matrices for each batch.

    Returns:
        list of Tensor: Transformed points in the camera frame, with the same length as lid_points.
    """
    batch_size = len(lid_points)
    projected_lid_points = []

    for b in range(batch_size):
        num_points = lid_points[b].shape[0]  # Number of points for the current batch
        ones = torch.ones((num_points, 1), device=lid_points[b].device, dtype=lid_points[b].dtype)
        
        # Convert points to homogeneous coordinates
        lid_points_h = torch.cat((lid_points[b], ones), dim=-1)  # Shape [N, 3]

        # Perform the transformation
        projected_h = torch.matmul(lid_points_h, homography_matrix[b].T)  # Shape [N, 3]

        # Normalize to get 2D points
        projected_points = projected_h[:, :2] / projected_h[:, 2:3]  # Shape [N, 2]

        # Append results
        projected_lid_points.append(projected_points)

    return projected_lid_points  

def compute_loss( img_points, projected_lid_points, lid_cents_proj_Org, img_h=1080, img_w=1440):
    """
    Compute the matching loss for a batch of inputs.

    Args:
        img_points (Tensor): Shape [B, N, 2], representing 2D points in the image.
        projected_lid_points (Tensor): Shape [B, M, 2], representing 2D points projected from LiDAR.

    Returns:
        loss (Tensor): A scalar loss value representing the batch's total loss.
    """
    batch_size = len(img_points)  # Extract batch size
    total_loss = 0.0  # Initialize total loss for the batch 
    total_loss_org = 0.0

    for b in range(batch_size):
        batch_loss = (torch.norm(projected_lid_points[b] - img_points[b], dim=1).mean()
                      )
        total_loss += (batch_loss)
        
        ##### loss with using org matrix
        batch_loss_org = (torch.norm(lid_cents_proj_Org[b] - img_points[b], dim=1).mean()
                      )
        total_loss_org += (batch_loss_org)

    # Average loss across the batch
    total_loss = total_loss / batch_size
    total_loss_org = total_loss_org / batch_size
    Diff = total_loss.item()-total_loss_org.item()
    print('\nNew Loss=',total_loss.item(),'| Org Loss=',total_loss_org.item(),'| Diff=',Diff)

    return total_loss, total_loss_org, Diff


if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    #---------------------------------#
    Cuda            = torch.cuda.is_available()
    #----------------------------------------------#
    seed            = 11
    #----------------------------------------------#
    token_len       = 256 #512 #256
    out_dim         = 9
    #----------------------------------------------#
    test_save_dir        = './test_FineTune'
    #---------------------------------------------------------------------#
    classes_path    = './coco_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = './best_weights/best_epoch_weights.pth'
    #------------------------------------------------------#
    image_shape     = [1080, 1440]
    input_shape     = [384, 384]  #PyTorch--> [C, H, W]; CiFar dataset
    #------------------------------------------------------------------#
    label_smoothing     = 0
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 1 #4

    #------------------------------------------------------------------#
    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    optimizer_type      = "Adam" #"sgd"
    momentum            = 0.937
    weight_decay        = 0 #5e-4
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    num_workers         = 0
    #------------------------------------------------------------------#
    pretrain_vit_path   = './pretrained_weights/jx_vit_base_p16_384-83fb41ba.pth'

    #------------------------------------------------------#
    seed_everything(seed)
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #------------------------------------------------------------------#
    homography_matrix   = np.load(r'./homography_matrix_lid2cam_best_init.npy', allow_pickle=True) 
    if TEST:
        new_matrix   = np.load(test_save_dir + '/new_H.npy', allow_pickle=True)
    #------------------------------------------------------------------#
    max_dis_cost        = (image_shape[0] + image_shape[1])/16.0 #5.0  #lid_cam_match_cost_thres
    



    #------------------------------------------------------#
    #   Get classes
    #------------------------------------------------------#
    #class_names, num_classes = get_classes(classes_path)
  
    ###################    Dataset  ######################  
    # #---------------------------#
    # #   Generate train_data file paths TXT 
    # #---------------------------#
    dataset_dir = r'/xdisk/caos/leicheng/FHWA/Lei_new_dataset_FineTune'
    train_data_file = '/xdisk/caos/leicheng/FHWA/train_newDataset_paths_FineTune.txt'    
    # save_sorted_file_paths(dataset_dir, train_data_file)
    
    #---------------------------#
    #   Split train and validation
    #---------------------------#
    train_set, val_set, test_set = split_data(train_data_file, N=90, M=10, X=90, Y=10)
    #val_set     = load_datapath(new_data_file)
    num_train   = len(train_set)
    num_val     = len(val_set) 
    num_test    = len(test_set)   
    # Shuffle the training data
    random.shuffle(train_set)  
    
    
    ##########################################  Our  Model   #########################        
    #------------------------------------------------------#
    #   Build Model
    #------------------------------------------------------#   
    '''FineTune_Model LOADING'''    
    if model_path != '':
        pretrain_vit_path = None
        
    model = FineTune_Model(pretrain_path = pretrain_vit_path, input_shape = input_shape, out_dim = out_dim, token_len = token_len)
    
    #------------------------------------------------------#
    #   Load weights
    #------------------------------------------------------#
    if model_path != '':
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))        
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        if len(no_load_key) > 0:
            print("\n\033[1;33;44mWarning: Backbone should be loaded successfully!\033[0m")
    
    #------------------------------------------------------#
    #   Use CUDA
    #------------------------------------------------------#
    model     = model.eval()   #test
    if Cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()     



    #------------------------------------------------------#
    #              Test  Here
    #------------------------------------------------------#
    batch_size = Unfreeze_batch_size
    
    #---------------------------------------#
    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size
    

    
    #---------------------------------------#
    #   DataLoader
    #---------------------------------------#
    test_dataset   = Dataset(test_set, input_shape, None, homography_matrix, epoch_length=UnFreeze_Epoch, max_dis_cost=max_dis_cost, \
                                    target_shape=None, interpolation='nearest', mode='val')

    
    train_sampler   = None
    val_sampler     = None
    shuffle         = False

    gen             = DataLoader(test_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=train_sampler, 
                                worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))


    #---------------------------------------#
    #   Start Testing
    #---------------------------------------#
    # Create the test_save directory if it does not exist
    os.makedirs(test_save_dir, exist_ok=True)
    save_path = test_save_dir + '/Test_DL_FT_results.txt'
    save_H = test_save_dir + '/new_H.npy'
    with open(save_path, 'w') as file:
        file.write("") #empty file
    
    local_rank = 0
    # Initialize counters for correct predictions and total samples
    loss        = 0
    new_loss    = 0
    org_loss    = 0
    loss_min    = 0
    num_samples = 0
    
    print('Start Testing')
    
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, lidars, img_cents, lid_cents, lid_cents_proj = batch
        # with torch.no_grad():
        #     if Cuda:
        #         images    = images.cuda(local_rank)
        #         lidars    = lidars.cuda(local_rank)
        #         # Move each tensor in the lidars list to the GPU
        #         #lidars = [lidar.cuda(local_rank) for lidar in lidars]
        #         img_cents = [a.cuda(local_rank) for a in img_cents]
        #         lid_cents = [a.cuda(local_rank) for a in lid_cents]
        #         lid_cents_proj = [a.cuda(local_rank) for a in lid_cents_proj]    

        #----------------------#
        #   Forward Propagation
        #----------------------#
        delta_matrix, _ = model(images, lidars, img_cents, lid_cents, lid_cents_proj)
        delta_matrix    = delta_matrix.cpu().detach()
        #----------------------#
        #   Compute Loss
        #----------------------#
        delta_matrix = delta_matrix.view(-1, 3, 3)
        homography_matrix = torch.tensor(homography_matrix, device=delta_matrix.device, dtype=delta_matrix.dtype)
        zeros_matrix = torch.zeros_like(delta_matrix)
        org_matrix = homography_matrix + zeros_matrix
        lid_cents_proj_Org = torch_lid2cam_proj(lid_cents, org_matrix)
        
        if not Intrinsic_Learning:
            ### Delta            
            #delta_matrix = delta_matrix.cpu().detach().numpy()
            new_matrix_with_off = homography_matrix + delta_matrix
            if TEST:
                new_matrix_with_off = torch.tensor(new_matrix, device=delta_matrix.device, dtype=delta_matrix.dtype)
            lid_cents_proj_New = torch_lid2cam_proj(lid_cents, new_matrix_with_off)
            #lid_cents_proj_New = torch_lid2cam_proj(lid_cents_proj, delta_matrix)
            ## Matrix Regularization loss: By penalizing large values in delta_H, the model learns to make only the necessary adjustments to initial H.
            mat_reg_loss = delta_matrix.abs().mean(dim=(1, 2))
            new_H = new_matrix_with_off
        else:
            ### Intrinsic
            intrinsic_matrix = delta_matrix
            new_H = torch.bmm(org_matrix, intrinsic_matrix)  # Shape: [B, 3, 3]
            if TEST:
                new_H = torch.tensor(new_matrix, device=delta_matrix.device, dtype=delta_matrix.dtype)
            lid_cents_proj_New = torch_lid2cam_proj(lid_cents, new_H)
            #lid_cents_proj_New = torch_lid2cam_proj(lid_cents_proj, intrinsic_matrix)
            H_identity = torch.eye(3, device=intrinsic_matrix.device)
            mat_reg_loss = (intrinsic_matrix - H_identity).abs().mean(dim=(1, 2))

        new_loss_value, org_loss_value, loss_value = compute_loss(img_cents, lid_cents_proj_New, lid_cents_proj_Org, img_h=1080, img_w=1440)
        
        loss += loss_value
        new_loss += new_loss_value
        org_loss += org_loss_value

        # Accumulate the number of samples
        num_samples = iteration + 1
        
        if not TEST:
            # Check if the current loss is smaller than the minimum loss
            if loss_value < loss_min:
                # Update the minimum loss value
                loss_min = loss_value
                
                # Save the current new_H as a .npy file
                np.save(save_H, new_H.cpu().detach().numpy())
                
                # Print a message indicating the new minimum loss and save operation
                print(f"Iteration: {iteration}: New mini_loss {loss_min} found. Saved new_H to {save_path}")
            
        # save results
        with open(save_path, 'a') as file:
            file.write(f"#####    Iteration: {iteration}   ##### \n")
            file.write(f"loss_value: {loss_value:.5f} \n")
            file.write(f"New Matrix: {new_H.cpu().detach().numpy()} \n")
            file.write(f"img_cents: {img_cents[0].cpu().detach().numpy()} \n")
            file.write(f"lid_cents_proj_New: {lid_cents_proj_New[0].cpu().detach().numpy()} \n")
            file.write(f"lid_cents_proj_Org: {lid_cents_proj_Org[0].cpu().detach().numpy()} \n")
            file.write("*********************************** \n")
        



    # Final accuracy calculations
    Final_loss = loss / num_samples

    
    print(f"Final Loss: {Final_loss:.3f}")

    
    # save results
    with open(save_path, 'a') as file:
        file.write(f"Final Loss: {Final_loss:.3f}")
        file.write(f"new_loss: {new_loss / num_samples:.3f}")
        file.write(f"org_loss: {org_loss / num_samples:.3f}")


    print(f"DL_FineTune results saved to {save_path}")