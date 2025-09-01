"""
Author: Lei Cheng
"""
import datetime
import os
from functools import partial

import numpy as np
import random
from scipy.spatial import distance
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils_finetune.utils import (ModelEMA, get_classes, seed_everything,
                         show_config, worker_init_fn, LossHistory)
from utils_finetune.utils_fit import fit_one_epoch

## For dataset
from utils_finetune.dataset_generator import Dataset, dataset_collate

################################################################################
from fine_tune_model import FineTune_Model
################################################################################
def save_sorted_file_paths(source_dir, destination_file):
    """
    Lists all files in the source directory, sorts them, 
    and saves their full paths into a TXT file.

    # Example usage:
    source_dir = '/xdisk/caos/leicheng/FHWA/Lei_new_dataset'
    destination_file = '/xdisk/caos/leicheng/FHWA/train_newDataset_paths.txt'    
    save_sorted_file_paths(source_dir, destination_file)
    """
    # Get the list of all files in the source directory and sort them
    file_paths = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            full_path = os.path.join(root, file)
            # Check if the file is non-empty
            if os.path.getsize(full_path) > 0:
                file_paths.append(full_path)
    # Sort the list of file paths
    file_paths.sort()
    # Write the sorted file paths to the TXT file
    with open(destination_file, 'w') as f:
        for path in file_paths:
            f.write(f"{path}\n")
    print(f"File paths saved to {destination_file}")
    
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

## Lei-Loss

class MatchingLoss(torch.nn.Module):
    """
    PyTorch-compatible loss function for matching 2D points based on a distance threshold.
    """
    def __init__(self, threshold=100, img_h=1080, img_w=1440, lambda_reg=0.01, simple_loss = True):
        """
        Initialize the MatchingLoss function.

        Args:
            threshold (float): Distance threshold to consider points as a match.
        """
        super(MatchingLoss, self).__init__()
        self.threshold = threshold
        self.img_h = img_h
        self.img_w = img_w
        self.simple_loss = simple_loss
        self.lambda_reg = lambda_reg
        
    def greedy_cost_matching(self, cost_matrix):
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
    
    def match_cam_to_lid(self, trackers, detections, max_dis_cost=100):
        """
        Match camera and LiDAR points based on a cost matrix and the greedy matching algorithm.
    
        Args:
            trackers (Tensor): Shape [N, 2], representing points from camera.
            detections (Tensor): Shape [M, 2], representing points from LiDAR.
            max_dis_cost (float): Maximum allowed distance for a valid match.
    
        Returns:
            matches (list): Matched indices [(row, col), ...].
            unmatched_detections (Tensor): Indices of unmatched LiDAR points.
            unmatched_trackers (Tensor): Indices of unmatched camera points.
        """
        cost_matrix = torch.cdist(trackers, detections, p=2)
        matches, row_ind, col_ind = self.greedy_cost_matching(cost_matrix)
    
        unmatched_trackers = torch.tensor([t for t in range(len(trackers)) if t not in row_ind], dtype=torch.int32, device=trackers.device)
        unmatched_detections = torch.tensor([d for d in range(len(detections)) if d not in col_ind], dtype=torch.int32, device=detections.device)
    
        # Filter matches based on distance threshold
        filtered_matches = []
        for t_idx, d_idx in matches:
            if cost_matrix[t_idx, d_idx] < max_dis_cost:
                filtered_matches.append((t_idx, d_idx))
            else:
                unmatched_trackers = torch.cat([unmatched_trackers, torch.tensor([t_idx], device=trackers.device)])
                unmatched_detections = torch.cat([unmatched_detections, torch.tensor([d_idx], device=detections.device)])

        # Convert filtered_matches to Tensor
        if filtered_matches:
            filtered_matches = torch.tensor(filtered_matches, dtype=torch.int32, device=trackers.device)
        else:
            filtered_matches = torch.empty((0, 2), dtype=torch.int32, device=trackers.device)

        return filtered_matches, unmatched_detections, unmatched_trackers
    
    def forward_Match(self, img_points, projected_lid_points, projected_lid_points_Org):
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

        for b in range(batch_size):
            if not self.simple_loss:
                # Perform matching
                matches, unmatched_lids, unmatched_cams = self.match_cam_to_lid(
                    img_points[b], projected_lid_points[b], max_dis_cost=self.threshold
                )
    
                # Compute match loss
                if matches.numel() > 0:
                    match_loss = torch.mean(torch.tensor([
                        torch.norm(img_points[b][m[0]] - projected_lid_points[b][m[1]])
                        for m in matches
                    ], device=img_points[b].device))
                else:
                    match_loss = self.threshold * 1.2
    
                # Compute unmatched penalty
                unmatched_count = len(unmatched_lids) + len(unmatched_cams)
                unmatched_penalty = unmatched_count / (img_points[b].size(0) + projected_lid_points[b].size(0))
    
                unmatched_loss = 0
                if unmatched_lids.numel() > 0 and unmatched_cams.numel() > 0:
                    unmatched_loss = torch.norm(
                        torch.mean(projected_lid_points[b][unmatched_lids], dim=0) -
                        torch.mean(img_points[b][unmatched_cams], dim=0)
                    ) / max(self.img_h, self.img_w)
    
                # Combine match loss and unmatched penalty
                batch_loss = match_loss + unmatched_penalty * self.threshold + unmatched_loss * self.threshold
            else:
                batch_loss = torch.norm(projected_lid_points[b] - projected_lid_points_Org[b], dim=1).mean() / max(self.img_h, self.img_w)
            
            total_loss += batch_loss

        # Average loss across the batch
        total_loss /= batch_size
        print('- Lei-Total Loss=',total_loss.item())
        return total_loss   

    def forward_simple(self, img_points, projected_lid_points, mat_reg_loss, lid_cents_proj_Org):
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
        total_loss_final = 0.0

        for b in range(batch_size):
            batch_loss = (torch.norm(projected_lid_points[b] - img_points[b], dim=1).mean()
                          ) / max(self.img_h, self.img_w)
            #mat_reg_loss = mat_reg_loss / max(self.img_h, self.img_w) 
            total_loss += (batch_loss + self.lambda_reg * mat_reg_loss[b])
            
            ##### loss with using org matrix
            batch_loss_org = (torch.norm(lid_cents_proj_Org[b] - img_points[b], dim=1).mean()
                          ) / max(self.img_h, self.img_w)
            #mat_reg_loss = mat_reg_loss / max(self.img_h, self.img_w) 
            total_loss_org += (batch_loss_org + self.lambda_reg * mat_reg_loss[b])
            
            diff = batch_loss - batch_loss_org
            # Reward for negative diff
            reward = (1 + 5 * F.relu(-diff)) * diff
            # Penalty for positive diff
            penalty = 5 * F.relu(diff)
            # Combine terms with scaling
            total_loss_final += ( (reward + penalty) / (1 + torch.abs(diff).sum()) )
            #total_loss_final += ( 10* F.relu(diff))


        # Average loss across the batch
        total_loss = 1000* total_loss / batch_size
        total_loss_org = 1000* total_loss_org / batch_size
        print('\nTotal Loss=',total_loss.item(),'| Org Loss=',total_loss_org.item(),'| Diff=',total_loss.item()-total_loss_org.item())
        total_loss_final = 1000* total_loss_final / batch_size
        print('\nFinal Loss=',total_loss_final.item())
        return total_loss_final
    
    def forward(self, img_points, projected_lid_points, mat_reg_loss, lid_cents_proj_Org):
        total_loss = self.forward_simple(img_points, projected_lid_points, mat_reg_loss, lid_cents_proj_Org)
        return total_loss

    
 
    # def match_cam_to_lid(self, trackers, detections, max_dis_cost=100):
    #     """
    #     Match camera and LiDAR points based on a cost matrix and the greedy matching algorithm.
        
    #     Args:
    #         trackers (ndarray): Shape [N, 2], representing points from camera.
    #         detections (ndarray): Shape [M, 2], representing points from LiDAR.
    #         max_dis_cost (float): Maximum allowed distance for a valid match.
        
    #     Returns:
    #         matches (ndarray): Matched indices of shape [k, 2].
    #         unmatched_detections (ndarray): Indices of unmatched LiDAR points.
    #         unmatched_trackers (ndarray): Indices of unmatched camera points.
    #     """
    #     # Initialize 'cost_matrix'
    #     cost_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    
    #     # Populate 'cost_matrix' using Euclidean distances
    #     for t, tracker in enumerate(trackers):
    #         for d, detection in enumerate(detections):
    #             cost_matrix[t, d] = distance.euclidean(tracker, detection)
    
    #     # Produce matches using the greedy matching algorithm
    #     matches, row_ind, col_ind = self.greedy_cost_matching(cost_matrix)
    
    #     # Populate 'unmatched_trackers'
    #     unmatched_trackers = [t for t in range(len(trackers)) if t not in row_ind]
    
    #     # Populate 'unmatched_detections'
    #     unmatched_detections = [d for d in range(len(detections)) if d not in col_ind]
    
    #     # Finalize matches with threshold check
    #     filtered_matches = []
    #     for t_idx, d_idx in matches:
    #         if cost_matrix[t_idx, d_idx] < max_dis_cost:
    #             filtered_matches.append([t_idx, d_idx])
    #         else:
    #             unmatched_trackers.append(t_idx)
    #             unmatched_detections.append(d_idx)
    
    #     # Return results
    #     return np.array(filtered_matches), np.array(unmatched_detections), np.array(unmatched_trackers)
            

    # def forward(self, img_points, projected_lid_points):
    #     """
    #     Compute the matching loss for a batch of inputs.

    #     Args:
    #         img_points (Tensor): Shape [B, N, 2], representing 2D points in the image.
    #         projected_lid_points (Tensor): Shape [B, M, 2], representing 2D points projected from LiDAR.

    #     Returns:
    #         loss (Tensor): A scalar loss value representing the batch's total loss.
    #     """
    #     batch_size = len(img_points)  # Extract batch size
    #     total_loss = 0.0  # Initialize total loss for the batch 
        
    #     for b in range(batch_size):
    #         # Extract points for the current batch item
    #         img_points_np = img_points[b].cpu().detach().numpy()
    #         projected_lid_points_np = projected_lid_points[b].cpu().detach().numpy()

    #         # Perform matching
    #         matches, unmatched_lids, unmatched_cams = self.match_cam_to_lid(
    #             img_points_np, projected_lid_points_np, max_dis_cost=self.threshold
    #         )

    #         # Compute match loss
    #         if matches.size > 0:
    #             match_loss = (
    #                 torch.tensor(
    #                     [
    #                         distance.euclidean(img_points_np[m[0]], projected_lid_points_np[m[1]])
    #                         for m in matches
    #                     ],
    #                     device=img_points[b].device,
    #                 ).mean()
    #             )
    #         else:
    #             match_loss = torch.tensor(self.threshold * 1.2, device=img_points[b].device)

    #         # Compute unmatched penalty
    #         unmatched_count = len(unmatched_lids) + len(unmatched_cams)
    #         unmatched_penalty = unmatched_count / (
    #             img_points[b].shape[0] + projected_lid_points[b].shape[0]
    #         )
            
    #         unmatched_loss = abs(np.mean([projected_lid_points_np[u] for u in unmatched_lids]) - \
    #                             np.mean([img_points_np[u] for u in unmatched_cams]))/ max(self.img_h, self.img_w)

    #         # Combine match loss and unmatched penalty
    #         batch_loss = match_loss + torch.tensor(unmatched_penalty * self.threshold, device=img_points[b].device) + \
    #                             torch.tensor(unmatched_loss * self.threshold, device=img_points[b].device)

    #         # Accumulate loss
    #         total_loss += batch_loss

    #     # Average loss across the batch
    #     total_loss /= batch_size    
    #     return torch.tensor(total_loss, device='cuda', requires_grad=True)








    


if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    #---------------------------------#
    Cuda            = torch.cuda.is_available()
    #----------------------------------------------#
    seed            = 11
    #----------------------------------------------#
    token_len       = 256 #512 #256
    out_dim         = 9
    #---------------------------------------------------------------------#
    classes_path    = './coco_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = '' #'' #'best_model/best_epoch_weights.pth'
    #------------------------------------------------------#
    image_shape     = [1080, 1440]
    input_shape     = [384, 384]  #PyTorch--> [C, H, W]; CiFar dataset
    #------------------------------------------------------------------#
    label_smoothing     = 0
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 800 #300 #100
    Unfreeze_batch_size = 8#4 #12 #16

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
    #------------------------------------------------------------------#
    max_dis_cost        = (image_shape[0] + image_shape[1])/ 16.0 #5.0  #lid_cam_match_cost_thres
    


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
        
    model = FineTune_Model(pretrain_path = pretrain_vit_path, input_shape = input_shape, out_dim = 9, token_len = token_len)
    
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
        print("\n\033[1;33;44mWarning: Backbone should be loaded successfully!\033[0m")
    
    #------------------------------------------------------#
    #   Use CUDA
    #------------------------------------------------------#
    model     = model.train()
    if Cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()     
        
    #------------------------------------------------------#
    #   Set Loss function
    #------------------------------------------------------#
    loss_func = MatchingLoss(threshold=100, img_h=image_shape[0], img_w=image_shape[1], lambda_reg=0.001)

    
    
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    #---------------------------------------#
    #   optimizer
    #---------------------------------------#
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Init_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=5e-4
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=Init_lr, momentum=0.9, weight_decay=5e-4)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=Init_lr,
                          momentum=0.9, weight_decay=5e-4)    
    #---------------------------------------#
    #   lr_scheduler
    #---------------------------------------#
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) 
     
            
    #----------------------------#
    #   Weights Smoothing
    #----------------------------#
    ema = ModelEMA(model)
    

    #----------------------------#
    #  Show Config
    #----------------------------#
    show_config(
        classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = None, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = None, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = None, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    #------------------------------------------------------#
    #   
    #------------------------------------------------------#
    if True:
        batch_size = Unfreeze_batch_size
        
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        

        if ema:
            ema.updates     = epoch_step * Init_Epoch
        
        #---------------------------------------#
        #   DataLoader
        #---------------------------------------#
        train_dataset   = Dataset(train_set,input_shape, None, homography_matrix, epoch_length=UnFreeze_Epoch, max_dis_cost=max_dis_cost, \
                                        target_shape=None, interpolation='nearest', mode='train')
        val_dataset     = Dataset(val_set, input_shape, None, homography_matrix, epoch_length=UnFreeze_Epoch, max_dis_cost=max_dis_cost, \
                                        target_shape=None, interpolation='nearest', mode='val')
        
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        #---------------------------------------#
        #   Start Training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            print('epoch',epoch,'\n')
            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch
           
            fit_one_epoch(model, ema, loss_func, homography_matrix, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, save_period, save_dir)
            
            # Update the learning rate
            scheduler.step()

        loss_history.writer.close()
