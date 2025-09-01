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

from utils.utils import (ModelEMA, get_classes, seed_everything,
                         show_config, worker_init_fn, LossHistory)
from utils.utils_fit import fit_one_epoch

## For dataset
from utils.dataset_generator import Dataset, dataset_collate

################################################################################
from ResNet import ResNet50
from discriminator import Comm_Feat_Discriminator
################################################################################

def plot_and_save_results(images, lidars, img_clss, lid_clss, img_cls_preds, lid_cls_preds, labels, binary_preds, class_names, iteration, save_dir):
    """
    Plots the image and lidar data with their corresponding real and predicted classifications,
    and saves the plot to the specified directory.

    Args:
        images (torch.Tensor): The images tensor.
        lidars (torch.Tensor): The lidar data tensor.
        img_clss (torch.Tensor): The real image class.
        lid_cls_preds (torch.Tensor): The predicted lidar class.
        lid_clss (torch.Tensor): The real lidar class.
        labels (torch.Tensor): The real binary labels.
        binary_preds (torch.Tensor): The predicted binary labels.
        class_names (list): The list of class names.
        iteration (int): The current iteration number for naming the files.
        save_dir (str): The directory where the plot and results will be saved.
    """


    batch_size = images.size(0)
    
    for i in range(batch_size):
        # Extract individual data from the batch
        img = images[i].cpu()
        lidar = lidars[i].cpu()
        img_cls = img_clss[i].cpu()
        img_cls_pred = img_cls_preds[i].cpu()
        lid_cls_pred = lid_cls_preds[i].cpu()
        lid_cls = lid_clss[i].cpu()
        label = labels[i].cpu()
        binary_pred = binary_preds[i].cpu()
        
        # Calculate the unique file identifier based on the iteration and item index in the batch
        file_index = iteration * batch_size + i

        # Plot the image on the left
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img.permute(1, 2, 0).numpy().astype(np.uint8))  # Assuming images are in [C, H, W] format
        plt.title(f"True: {class_names[img_cls.item()]} | Pred: {class_names[img_cls_pred.item()]}")
        plt.axis('off')

        # Plot the lidar data on the right
        plt.subplot(1, 2, 2)
        plt.scatter(lidar[:, 0].numpy(), lidar[:, 1].numpy(), alpha=0.6, edgecolors='w', linewidth=0.5)
        plt.title(f"True: {class_names[lid_cls.item()]} | Pred: {class_names[lid_cls_pred.item()]}")
        # plt.xlabel("Lidar X")
        # plt.ylabel("Lidar Y")
        # plt.grid(True)
        plt.axis('off')

        # Set the overall title based on binary classification
        true_label = 'same' if label.int().item() != 0 else 'differ'
        pred_label = 'same' if binary_pred.int().item() != 0 else 'differ'
        plt.suptitle(f"True: {true_label} vs Pred: {pred_label}", fontsize=16)

        # Save the plot with the filename as the iteration number and batch index
        plot_save_path = os.path.join(save_dir, f"{file_index}.png")
        plt.savefig(plot_save_path)
        plt.close()

    print(f"{file_index}.png saved to {save_dir}")


    
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

if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    #---------------------------------#
    Cuda            = torch.cuda.is_available()
    #----------------------------------------------#
    seed            = 11
    #----------------------------------------------#
    token_len       = 256 #512 #256
    #----------------------------------------------#
    test_save_dir        = './test_plots'
    #---------------------------------------------------------------------#
    classes_path    = './coco_classes.txt'
    #train_data_file = '/xdisk/caos/leicheng/FHWA/train_newDataset_paths.txt'
    train_data_file = '/xdisk/caos/leicheng/FHWA/train_Dataset3_paths.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = './best_model/best_epoch_weights.pth'
    #------------------------------------------------------#
    input_shape     = [32, 32]  #PyTorch--> [C, H, W]; CiFar dataset
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

    #------------------------------------------------------#
    seed_everything(seed)
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #------------------------------------------------------#
    #   Get classes
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
  
    ###################    Dataset  ######################  
    #---------------------------#
    #   Split train and validation
    #---------------------------#
    train_set, val_set, test_set = split_data(train_data_file, N=90, M=10, X=90, Y=10)
    num_train   = len(train_set)
    num_val     = len(val_set) 
    num_test    = len(test_set)   
    # Shuffle the training data
    random.shuffle(train_set)    
    
    
    ##########################################  Our  Model   #########################        
    #------------------------------------------------------#
    #   Build Model
    #------------------------------------------------------#  
    '''Discriminator MODEL LOADING'''    
    model = Comm_Feat_Discriminator(num_class=num_classes, token_len = token_len)
    
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
    test_dataset   = Dataset(test_set, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                    target_shape=[256,256], interpolation='nearest', mode='test')

    
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
    
    local_rank = 0
    # Initialize counters for correct predictions and total samples
    binary_correct_total = 0
    img_cls_correct_total = 0
    lid_cls_correct_total = 0
    num_samples = 0
    
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, lidars, img_cents, lid_cents, labels, img_clss, lid_clss = batch
        with torch.no_grad():
            if Cuda:
                images    = images.cuda(local_rank)
                # Move each tensor in the lidars list to the GPU
                #lidars = [lidar.cuda(local_rank) for lidar in lidars]
                lidars    = lidars.cuda(local_rank)
                img_cents = img_cents.cuda(local_rank)
                lid_cents = lid_cents.cuda(local_rank)
                labels    = labels.cuda(local_rank)
                img_clss  = img_clss.cuda(local_rank)
                lid_clss  = lid_clss.cuda(local_rank)

        #----------------------#
        #   Forward Propagation
        #----------------------#
        outputs, img_cls_res, lid_cls_res, l3_points = model(images, lidars, img_cents, lid_cents)
        
        # Calculate binary classification accuracy
        binary_preds = torch.sigmoid(outputs) > 0.5
        binary_correct_total += (binary_preds.int() == labels.int()).sum().item()

        # Calculate multi-class classification accuracy for image and lidar
        _, img_cls_preds = torch.max(img_cls_res, dim=1)
        _, lid_cls_preds = torch.max(lid_cls_res, dim=1)

        img_cls_correct_total += (img_cls_preds == img_clss).sum().item()
        lid_cls_correct_total += (lid_cls_preds == lid_clss).sum().item()

        # Accumulate the number of samples
        num_samples += labels.size(0)
        
        # Save Test Plots
        plot_and_save_results(images, lidars, img_clss, lid_clss, img_cls_preds, lid_cls_preds, labels, 
                              binary_preds, class_names, iteration, test_save_dir)


    # Final accuracy calculations
    binary_accuracy = binary_correct_total / num_samples
    img_cls_accuracy = img_cls_correct_total / num_samples
    lid_cls_accuracy = lid_cls_correct_total / num_samples
    
    print(f"Binary Classification Accuracy: {binary_accuracy * 100:.2f}%")
    print(f"Image Classification Accuracy: {img_cls_accuracy * 100:.2f}%")
    print(f"Lidar Classification Accuracy: {lid_cls_accuracy * 100:.2f}%")
    
    # save results
    save_path = './accuracy_results.txt'
    with open(save_path, 'w') as file:
        file.write(f"Binary Classification Accuracy: {binary_accuracy * 100:.2f}%\n")
        file.write(f"Image Classification Accuracy: {img_cls_accuracy * 100:.2f}%\n")
        file.write(f"Lidar Classification Accuracy: {lid_cls_accuracy * 100:.2f}%\n")

    print(f"Accuracy results saved to {save_path}")