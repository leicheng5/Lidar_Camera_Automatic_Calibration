"""
Author: Lei Cheng
"""
import datetime
import os
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

if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    #---------------------------------#
    Cuda            = torch.cuda.is_available()
    #----------------------------------------------#
    seed            = 11
    #----------------------------------------------#
    token_len       = 256 #512 #256
    #---------------------------------------------------------------------#
    classes_path    = './coco_classes.txt'
    train_data_file = '/xdisk/caos/leicheng/FHWA/train_Dataset3_paths.txt'
    #new_data_file = '/xdisk/caos/leicheng/FHWA/train_newDataset_paths_half.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'best_model/best_epoch_weights.pth' #'' #'best_model/best_epoch_weights.pth'
    #------------------------------------------------------#
    input_shape     = [32, 32]  #PyTorch--> [C, H, W]; CiFar dataset
    #------------------------------------------------------------------#
    label_smoothing     = 0
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 300 #100
    Unfreeze_batch_size = 4 #12 #16

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
    # only need to run once
    # source_dir = '/xdisk/caos/leicheng/FHWA/Lei_dataset3' 
    # save_sorted_file_paths(source_dir, train_data_file)
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
    #model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)
    
    '''Discriminator MODEL LOADING'''    
    model = Comm_Feat_Discriminator(num_class=num_classes, token_len = token_len)
    
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
    #loss_func = torch.nn.BCELoss()
    # Instantiate BCEWithLogitsLoss (no need to apply sigmoid manually)
    loss_func = torch.nn.BCEWithLogitsLoss()
    # classifi loss
    loss_func_cls = torch.nn.NLLLoss() #nn.CrossEntropyLoss()
    
    
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
        train_dataset   = Dataset(train_set,input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        target_shape=[256,256], interpolation='nearest', mode='train')
        val_dataset     = Dataset(val_set, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        target_shape=[256,256], interpolation='nearest', mode='val')
        
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
           
            fit_one_epoch(model, ema, loss_func, loss_func_cls, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, save_period, save_dir)
            
            # Update the learning rate
            scheduler.step()

        loss_history.writer.close()
