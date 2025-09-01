import pickle
from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from utils_finetune.utils import cvtColor, preprocess_input,resize_image
import random
from scipy.spatial import distance

class Dataset(Dataset):
    def __init__(self, train_sequences, input_shape, num_classes, homography_matrix, epoch_length, max_dis_cost=150, \
                       target_shape=[256,256,256], interpolation='nearest', mode='train', proj_lid=True):
        """ 
        Generate batch size of train data.
        """
        super(Dataset, self).__init__()
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.epoch_length       = epoch_length
        
        ## radar
        self.train_sequences    = train_sequences #annotation_lines
        self.length             = len(self.train_sequences)
        self.mode               = mode
        self.target_shape       = target_shape
        self.interpolation      = interpolation
        self.homography_matrix  = homography_matrix
        self.max_dis_cost           = max_dis_cost
        self.proj_lid           = proj_lid

    def __len__(self):
        return self.length
    
    ############  Lidar-Camera Pre-Matching  #####################################
    def lid2cam_proj(self, img_points, lid_points, homography_matrix):
        #matrix_path = os.path.join(output_path, r'homography_matrix_lid2cam_best.npy')
        #homography_matrix = np.load(matrix_path, allow_pickle=True)
        
        # Apply the homography perspective transformation using homography_matrix
        projected_lid_points = cv2.perspectiveTransform(lid_points.reshape(-1, 1, 2), homography_matrix)
        return projected_lid_points.squeeze(1)

    def filter_points_within_img(self, lid_cent_proj, lid_cent, img_cent, width=1440, height=1080):
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
        valid_mask = (lid_cent_proj[:, 0] >= -5) & (lid_cent_proj[:, 0] < (width+5)) & \
                     (lid_cent_proj[:, 1] >= -5) & (lid_cent_proj[:, 1] < (height+5))
        # valid_mask = (lid_cent_proj[:, 0] >= 0) & (lid_cent_proj[:, 0] < width) & \
        #              (lid_cent_proj[:, 1] >= 0) & (lid_cent_proj[:, 1] < height)                     
        
        # Filter all points using the mask
        filtered_lid_cent = lid_cent[valid_mask]
        filtered_img_cent = img_cent[valid_mask]
        filtered_lid_cent_proj = lid_cent_proj[valid_mask]
    
        return filtered_lid_cent_proj, filtered_lid_cent, filtered_img_cent

    
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
        matches, row_ind, col_ind = self.greedy_cost_matching(cost_matrix)
    
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

    def lid_cam_Association(self, proj_lidar_ctd, lidar_ctd, cam_ctd, max_dis_cost):
        # Match lidar to camera, set max_dis_cost per your need
        matched, unmatched_lidar, unmatched_camera = \
            self.match_cam_to_lid(cam_ctd, proj_lidar_ctd, max_dis_cost) 
       
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
                
    ############  Data Loader  #####################################
    def Data_Generator(self,frame_idx):
        """
        Generate train data(Img and GT_BBox) with batch size
        """
        #count = 0
        #while  count < len(self.RAD_sequences_train):
        train_filename = self.train_sequences[frame_idx] 
        # ## load radar RAD
        with open(train_filename, 'rb') as file:
            data = pickle.load(file)
        
        ### Parse data ###
        image    = data['image']
        lidar    = data['lidar']
        img_cent = data['img_cent']
        lid_cent = data['lid_cent']
        img_box  = data['cam_box']
        lid_box  = data['lid_box']

        return image, np.array(lidar), np.array(img_cent), np.array(lid_cent)


    def __getitem__(self, index):
        index       = index % self.length
        image, lidar, img_cent, lid_cent = self.Data_Generator(index) #[H, W, C]
        
        
        # Check if data is invalid
        if img_cent is None or lid_cent is None or len(img_cent) == 0 or len(lid_cent) == 0:
            # Sample a new random index
            new_index = random.randint(0, self.length - 1)
            return self.__getitem__(new_index)
        
        lid_cent = lid_cent[:,:2]
        
        # project lidar
        lid_cent_proj = None
        if self.proj_lid:
            lid_cent_proj = self.lid2cam_proj(img_cent, lid_cent, self.homography_matrix)
        
        lid_cent_proj, lid_cent, img_cent = self.lid_cam_Association(lid_cent_proj, lid_cent, img_cent, self.max_dis_cost)
        lid_cent_proj, lid_cent, img_cent = self.filter_points_within_img(lid_cent_proj, lid_cent, img_cent, width=1440, height=1080)
        
        # Check if data is invalid
        if img_cent is None or len(img_cent) == 0:
            # Sample a new random index
            new_index = random.randint(0, self.length - 1)
            return self.__getitem__(new_index)
        
        image    = resize_image(image, self.input_shape, letterbox_image=False)  
        image_np = np.transpose(np.array(image), (2, 0, 1)) # pytorch-->[C, H, W]
        return image_np, lidar, img_cent, lid_cent, lid_cent_proj



    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    
def pad_and_stack_tensors(tensor_batch, pad_value=0):
    """
    Pads a batch/list of tensors to have the same shape and stacks them into a single tensor.

    Args:
        tensor_batch (list of torch.Tensor): List of tensors to be padded and stacked.

    Returns:
        torch.Tensor: A single tensor with all input tensors stacked along a new dimension.
    """
    # Find the maximum shape along each dimension
    max_shape = [max(tensor.size(dim) for tensor in tensor_batch) for dim in range(len(tensor_batch[0].shape))]

    # Pad tensors to have the same shape
    padded_tensors = [
        F.pad(tensor, 
              (0, max_shape[1] - tensor.shape[1], 
               0, max_shape[0] - tensor.shape[0]), 
              value=pad_value) 
        for tensor in tensor_batch
    ]
    # Stack the padded tensors
    stacked_tensor = torch.stack(padded_tensors)

    return stacked_tensor

    
# For DataLoader's collate_fn, used to orgnize batch data
def dataset_collate(batch):
    images     = []
    lidars     = []
    img_cents  = []
    lid_cents  = []
    lid_cents_proj  = []

    for i, (image, lidar, img_cent, lid_cent, lid_cent_proj) in enumerate(batch):
        images.append(image)
        lidars.append(torch.from_numpy(lidar).type(torch.FloatTensor))
        img_cents.append(img_cent)
        lid_cents.append(lid_cent)
        lid_cents_proj.append(lid_cent_proj)


    images     = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    img_cents  = [torch.tensor(array, dtype=torch.float32) for array in img_cents]
    lid_cents  = [torch.tensor(array, dtype=torch.float32) for array in lid_cents]
    lid_cents_proj  = [torch.tensor(array, dtype=torch.float32) for array in lid_cents_proj]
    
    ### Pad lidar data ##
    lidars = pad_and_stack_tensors(lidars)

    return images, lidars, img_cents, lid_cents, lid_cents_proj


