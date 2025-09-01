#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""
############### Lei ###################################
import os, glob, shutil, sys
#sys.path.append("../") # add search path: sys.path.append("../../")
#######################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc
from moviepy.editor import ImageSequenceClip


def get_sorted_paths(directory, extension):
    files = glob.glob(os.path.join(directory, f"*.{extension}"))
    #sorted_paths = sorted([os.path.abspath(file) for file in files])
    sorted_paths = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return sorted_paths

def read_sorted_npy_files(sorted_paths, num=float('inf')):
    data = []
    for i, file_path in enumerate(sorted_paths):
        if i >= num:  # Check if the number of files read reaches the specified limit
            break
        # Load .npy file data using NumPy
        npy_data = np.load(file_path, allow_pickle=True)
        data.append(npy_data)
    return data

def read_sorted_png_files(sorted_paths, num=float('inf')):
    data = []
    for i, file_path in enumerate(sorted_paths):
        if i >= num:  # Check if the number of files read reaches the specified limit
            break
        # Read .png file using PIL (Pillow) library
        png_image = Image.open(file_path)
        # Convert image data to NumPy array
        png_data = np.array(png_image)
        data.append(png_data)
    return data

def get_sorted_files(directory, extension):
    files = glob.glob(os.path.join(directory, f"*.{extension}"))
    sorted_files = sorted(files)
    return sorted_files

def read_npy_files(directory,num):
    # Get sorted list of .npy files in the directory
    npy_files = get_sorted_files(directory, 'npy')
    data = []
    for i, file in enumerate(npy_files):
        if i >= num:  # Check if the number of files read reaches the specified limit
            break
        file_path = os.path.join(directory, file)
        # Load .npy file data using NumPy
        npy_data = np.load(file_path,allow_pickle=True)
        data.append(npy_data)
    return data

def read_png_files(directory,num):
    # Get sorted list of .png files in the directory
    png_files = get_sorted_files(directory, 'png')
    data = []
    for i, file in enumerate(png_files):
        if i >= num:  # Check if the number of files read reaches the specified limit
            break
        file_path = os.path.join(directory, file)
        # Read .png file using PIL (Pillow) library
        png_image = Image.open(file_path)
        # Convert image data to NumPy array
        png_data = np.array(png_image)
        data.append(png_data)
    return data

def get_subdirectories(parent_folder):
    subdirectories = []
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    return subdirectories

def filter_folders_with_names(folder_list, names):
    filtered_folders = []
    for folder_path in folder_list:
        folder_name = os.path.basename(folder_path)
        if folder_name in names:
            filtered_folders.append(folder_path)
    return filtered_folders

def filter_files_with_names(file_paths, names, mode='filter_file_name'):
    filtered_files = []
    for file_path in file_paths:
        if mode=='filter_file_name':
            # Check if the file_name contains the specified string
            file_name = os.path.basename(file_path)
            if any(filter_str in file_name for filter_str in names):
                filtered_files.append(file_path)
        elif mode=='filter_file_path':
            # Check if any of the filter strings exist in the complete file path
            if any(filter_str in file_path for filter_str in names):
                filtered_files.append(file_path)
    return filtered_files

# def calculate_bbox_centers(bbox_array):
#     centers = np.zeros((bbox_array.shape[0], 2))  # Initialize array to store centers
#     centers[:, 0] = (bbox_array[:, 0] + bbox_array[:, 2]) / 2  # Calculate x-coordinate of centers
#     centers[:, 1] = (bbox_array[:, 1] + bbox_array[:, 3]) / 2  # Calculate y-coordinate of centers
#     return centers

def calculate_bbox_centers(bbox_list):
    centers = []
    for bbox in bbox_list:
        xmin, ymin, xmax, ymax = np.transpose(bbox)
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center = np.column_stack((center_x, center_y))
        centers.append(center)
    return centers

def crop_3d_data(data, xmin, xmax, ymin, ymax):
    cropped_data = data[ymin:ymax, xmin:xmax, :]
    return cropped_data

def crop_data_for_bboxes(data_list, bbox_list):
    cropped_data_list = []
    for i in range(len(data_list)):
        data = data_list[i]
        bbox = bbox_list[i]
        
        # Check if bbox contains multiple bounding boxes
        if bbox.ndim == 1:
            bbox = bbox.reshape(1, -1)
        
        # Initialize list to store cropped data for each bounding box
        cropped_data_bbox = []
        
        # Crop data for each bounding box
        for single_bbox in bbox:
            xmin, ymin, xmax, ymax = np.transpose(single_bbox)
            cropped_data = crop_3d_data(data, int(xmin), int(xmax), data.shape[0] - int(ymax), data.shape[0] - int(ymin))
            cropped_data_bbox.append(cropped_data)
        
        # Append cropped data for current data and bbox to the main list
        cropped_data_list.append(cropped_data_bbox)
    
    return cropped_data_list



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
    if any(img_points):
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
  
def plot_paired_points_all(img_fr, paired_ctds_fr, i, path, matrix, mode='homography'):
    if not any(paired_ctds_fr):
        print("No points to plot. Skipping...")
        return
    # Convert the paired points to arrays
    paired_ctds_fr = np.array(paired_ctds_fr)
    img_points = paired_ctds_fr[:, 0]  # Image points
    radar_points = paired_ctds_fr[:, 1]  # Radar points
    if len(matrix)==3: #mode=='homography':
        # Apply the perspective transformation using homography_matrix
        transformed_points = cv2.perspectiveTransform(radar_points.reshape(-1, 1, 2), matrix)
    elif len(matrix)==2:    
        # Apply the affine transformation using affine_matrix
        transformed_points = cv2.transform(radar_points.reshape(-1, 1, 2), matrix)
        

    
    # Plot the image
    plt.imshow(img_fr)

    # Plot the image points
    plt.scatter(img_points[:, 0], img_points[:, 1], c='red', label='Image Points')

    # # Plot the radar points
    # plt.scatter(radar_points[:, 0], radar_points[:, 1], c='blue', label='Radar Points')
    
    # Plot the Projected Points
    plt.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='green', label='Projected Points')

    # Add legend
    #plt.legend()

    # Save the plot as an image file with a filename based on the iteration index
    filename = f'paired_points_plot_{i}.png'
    plt.savefig(path + filename)

    # Show the plot
    #plt.show()

def create_HQ_video_from_images(path, output_filename, fps=20):
    # Get the list of image files in the specified path
    image_files = glob.glob(os.path.join(path, '*.png'))
    
    # Sort the image files based on the numeric value in the filename
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    
    if not image_files:
        print("No image files found. Exiting...")
        return
    
    print(f"Number of images found: {len(image_files)}")
    print("First few image files:", image_files[:5])
    
    # Create a video clip from the image sequence
    clip = ImageSequenceClip(image_files, fps=fps)
    
    # Check the duration and fps of the clip
    print(f"Clip duration: {clip.duration}, fps: {clip.fps}")
    
    # Write the video file with high quality settings
    clip.write_videofile(output_filename, codec='libx264', fps=fps, bitrate="10000k", preset="slow", ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"Video created successfully: {output_filename}")
   
def create_video_from_images(path, output_filename):
    # Get the list of image files in the specified path
    image_files = glob.glob(path + '*.png')
    
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

def extract_data_from_txt(file_path):
    # Load the data from the file
    data = np.loadtxt(file_path, delimiter=',')
    
    # Split the data into LiDAR and camera arrays
    lidar_array = data[:, :3]
    camera_array = data[:, 3:5]
    
    return lidar_array, camera_array

###########################################################################
if __name__ == "__main__":
    path = r"E:\FHWA\calibration"
    file_path = os.path.join(path, r'picking_list1.txt')
    output_dir = r'E:\FHWA\calibration\outputs'
    matched_files_path = os.path.join(output_dir, r'matched_files.txt')
    save_path = os.path.join(path, r'result_images')
    input_cam_dir = r'E:\FHWA\camera_label\camera'    
    input_pcd_dir = r'E:\FHWA\lidar_pcds\lidar_pcds_1'
    input_nobg_dir = r'E:\FHWA\voxel_detector\outputs'
    input_cents_dir = r'E:\FHWA\voxel_detector\centroids'

    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    #############################  Load calibration matrix   #############################
    #homography_matrix = np.load(os.path.join(path, r'homography_matrix.npy'), allow_pickle=True)
    homography_matrix = np.load(os.path.join(path, r'homography_matrix_lid2cam.npy'), allow_pickle=True)
    
    #############################  Main   #############################
    # Load the matched files data
    data = np.loadtxt(matched_files_path, delimiter=',', dtype=str)
    # # Load the data from the npy file
    # lidar_array, camera_array = extract_data_from_txt(file_path)
    # # Define camera image coordinates and radar image coordinates
    # img_points   = camera_array
    # radar_points = lidar_array[:,:2]

    for i, entry in enumerate(data):
        print(f'Plot image {i}')
        # if entry[0].strip()!='1699029442.420000000.png':
        #     continue
        # Construct full paths for each file
        img_file = os.path.join(input_cam_dir, entry[0].strip())
        pcd_file = os.path.join(input_pcd_dir, entry[1].strip())
        nobg_pcd_file = os.path.join(input_nobg_dir, entry[2].strip())
        cent_file = os.path.join(input_cents_dir, entry[1].strip().rsplit('.', 1)[0] + '_centroids.npy')

        # Read the image file
        img_fr = plt.imread(img_file)
        
        # Read the centroids
        radar_points = np.load(cent_file, allow_pickle=True)
        radar_points = radar_points[:,:2]
        img_points   = []

        ### loop end

         
        plot_paired_points(img_fr, img_points, radar_points, i, save_path, homography_matrix)
            
        gc.collect()
                    
                    
    ####   END   #############################################################            
    # Create video
    output_filename = 'LC_paired_points.avi'
    create_video_from_images(save_path, output_filename)
    output_filename = 'LC_paired_points.mp4'
    create_HQ_video_from_images(save_path, output_filename)
    ###########################  Paired radar and image  #############################  

'''    
    
    ## Associating the radar and image by timestamps
    idx_rad,nearest_rad_time = find_nearest_betw_arr(rad_fr_timestamps, ts_img)
    
    #filtered_bbox_imgs_all will contain tuples with the index and corresponding non-empty data, preserving the original indices from bbox_imgs_all
    filtered_bbox_imgs_all = [(i, bbox) for i, bbox in enumerate(bbox_imgs_all) if bbox != [[]]]
    filtered_bbox_list = [bbox for _, bbox in filtered_bbox_imgs_all]
    filtered_idx_list  = [idx for idx,_ in filtered_bbox_imgs_all]
    
    # # Get the corresponding elements based on the filtered indices
    # filtered_all_imgs    = [all_imgs[i] for i in filtered_idx_list]
    # filtered_img_bbs_ctd = [img_bbs_ctd[i] for i in filtered_idx_list]
    # filtered_ts_img      = [ts_img[i] for i in filtered_idx_list]
    
    # ********************Delete unused variables to free up memory********************##############
    del bbox_imgs_all, filtered_bbox_imgs_all
    
    
    homography_matrix = np.load(path+'homography_matrix.npy', allow_pickle=True)
    affine_matrix     = np.load(path+'affine_matrix.npy', allow_pickle=True)
    
    # Get paired radar and image based on common feats
    paired_ctds = [] # List to store images and radar 's centers
    for i in range(len(filtered_bbox_list)):
    #for i in range(530, len(filtered_bbox_list)):
        print('img idx : ', i)
        # for each index i, obtain bbox_img_list and the corresponding bbox_rad_list
        bbox_img_list = filtered_bbox_list[i]
        
        # To store images and radar 's centers for one frame
        paired_ctds_fr = [] 
    
        # Get original index of the image
        img_idx = filtered_idx_list[i]
        # Get radar index based on the img_idx
        rad_idx = idx_rad[img_idx]
        # Get current radar path
        npy_path = npy_paths[rad_idx]
        png_path = png_paths[rad_idx]
        # Get current radar related data
        rad_bboxes, rad_centers, bbox_rads_all = process_radar_data(npy_path, png_path
                                                                    ,confidence=0.5, nms_iou=0.3
                                                                    ,yolo=radimg_yolo_model)
        
        bbox_rad_list   = bbox_rads_all[0]
        # Get rad_center_list
        rad_center_list = rad_centers[0]
        
        # Get corresponding data by index
        img_fr     = all_imgs[img_idx]
        bbs_ctd_fr = img_bbs_ctd[img_idx]
        #ts_img_fr  = ts_img[img_idx]   #rad_fr_timestamps[rad_idx]
        
        # to avoid repeat compare
        #uses the already_compared_same_j and already_compared_same_k variables to keep track 
        #of the indices that have already been compared and found to be the same. 
        #If either of these variables matches the current indices j and k, 
        #the loop continues to the next iteration without performing the comparison again.
        already_compared_same_j = None
        already_compared_same_k = None
    
        for j in range(len(bbox_img_list)):  # iterate over each img bbox in the bbox_img_list
            bbox_img = bbox_img_list[j]
            images = [bbox_img]
            
            # Check if already compared and found to be the same, then skip this iteration
            if already_compared_same_j == j:
                continue
            
            for k in range(len(bbox_rad_list)):  # iterate over each radar bbox in the bbox_rad_list
                # Check if already compared and found to be the same, then skip this iteration
                if already_compared_same_k == k:
                    continue
                
                bbox_rad = bbox_rad_list[k]
                rad_complexs = [bbox_rad]
    
                pred_labels = common_feats_find(images, rad_complexs,model=common_feats_model)
                print('same' if pred_labels[0] == 1 else 'diff')
                # plt.imshow(images[0])
                # plt.show()

                
                if pred_labels == 1:
                    bbs_ctd_img = bbs_ctd_fr[j]   # img bbox center
                    bbs_ctd_rad = list(rad_center_list[k])  # rad bbox center
                    paired_ctds.append([bbs_ctd_img, bbs_ctd_rad])
                    paired_ctds_fr.append([bbs_ctd_img, bbs_ctd_rad])
                    # Mark the current indices as already compared and found to be the same
                    already_compared_same_j = j
                    already_compared_same_k = k
        ### loop end

        save_path = path + 'result_images/'
        plot_paired_points(img_fr, paired_ctds_fr, i, save_path,homography_matrix)
            
        gc.collect()
                    
                    
    ####   END   #############################################################            
    # Save paired_ctds as an .npy file
    np.save(path+'paired_ctds.npy', np.array(paired_ctds))                    
        
    # Create video
    output_filename = 'paired_points_video.avi'
    create_video_from_images(path + 'result_images/', output_filename)
        

'''        

