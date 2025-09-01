import pickle
from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from utils.utils import cvtColor, preprocess_input,resize_image
## radar-Lei
import utils.radar_loader as loader
import utils.helper as helper


class Dataset(Dataset):
    def __init__(self, train_sequences, input_shape, num_classes, epoch_length, \
                       target_shape=[256,256,256], interpolation='nearest', mode='train'):
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

    def __len__(self):
        return self.length

        
    ############  RADData Loader  #####################################
    def Data_Generator(self,frame_idx):
        """
        Generate train data(Img and GT_BBox) with batch size
        """
        #count = 0
        #while  count < len(self.RAD_sequences_train):
        train_filename = self.train_sequences[frame_idx] 
        print(train_filename)
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


    def __getitem__(self, index):
        index       = index % self.length
        image, lidar, img_cent, lid_cent, label, img_cls, lid_cls = self.Data_Generator(index) #[H, W, C]

        image    = resize_image(image, self.input_shape, letterbox_image=False)  
        image_np = np.transpose(np.array(image), (2, 0, 1)) # pytorch-->[C, H, W]
        
        return image_np, lidar, img_cent, lid_cent, label, img_cls, lid_cls



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
    labels     = []
    img_clss   = []
    lid_clss   = []
    for i, (image, lidar, img_cent, lid_cent, label, img_cls, lid_cls) in enumerate(batch):
        images.append(image)
        lidars.append(torch.from_numpy(lidar).type(torch.FloatTensor))
        img_cents.append(img_cent)
        lid_cents.append(lid_cent)
        labels.append(label)
        img_clss.append(img_cls)
        lid_clss.append(lid_cls)

            
    images     = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    img_cents  = torch.from_numpy(np.array(img_cents)).type(torch.FloatTensor)
    lid_cents  = torch.from_numpy(np.array(lid_cents)).type(torch.FloatTensor)
    labels     = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)
    img_clss   = torch.from_numpy(np.array(img_clss)).type(torch.LongTensor)
    lid_clss   = torch.from_numpy(np.array(lid_clss)).type(torch.LongTensor) 
    
    # img_cents  = torch.tensor(img_cents).type(torch.FloatTensor)
    # lid_cents  = torch.tensor(lid_cents).type(torch.FloatTensor)
    # labels     = torch.tensor(labels).type(torch.FloatTensor)
    # img_clss   = torch.tensor(img_clss).type(torch.LongTensor)
    # lid_clss   = torch.tensor(lid_clss).type(torch.LongTensor) 
    
    ### Pad lidar data ##
    lidars = pad_and_stack_tensors(lidars)

    return images, lidars, img_cents, lid_cents, labels, img_clss, lid_clss


