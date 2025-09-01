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

def get_valid_point_indices(points, width, height):
    """
    Extracts the indices of points within the image boundaries.

    Args:
        transformed_points (numpy.ndarray): Array of points, shape (N, 2).
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        list: Indices of points that are within the image boundaries.
    """
    valid_indices = []

    # Iterate through the points and check if they are within the image boundaries
    for i, pt in enumerate(points):
        x, y = pt
        if 0 <= x < width and 0 <= y < height:
            valid_indices.append(i)

    return valid_indices


def split_points(img_points, lidar_points, width, height, Y_thr=None, X_thr=None):
    """
    Splits image and LiDAR points into different regions based on the provided Y_thr and X_thr.

    Args:
        img_points (numpy.ndarray): Array of image points with shape (N, 2), where N is the number of points.
        lidar_points (numpy.ndarray): Array of LiDAR points with shape (N, 3), where N is the number of points.
        width (int): Width of the image.
        height (int): Height of the image.
        Y_thr (float or int, optional): Y-coordinate threshold for vertical split. Default is None.
        X_thr (float or int, optional): X-coordinate threshold for horizontal split. Default is None.

    Returns:
        dict: A dictionary containing the split points. The keys depend on the presence of Y_thr and X_thr.
    """
    # Initialize the dictionary to hold the results
    split_data = {}

    if Y_thr is not None and X_thr is None:
        # Vertical 2-Split (Up/Down)
        up_idx = (img_points[:, 1] >= 0) & (img_points[:, 1] <= Y_thr)
        down_idx = (img_points[:, 1] > Y_thr) & (img_points[:, 1] <= height)

        split_data['img_points_1'] = img_points[up_idx]
        split_data['img_points_2'] = img_points[down_idx]
        split_data['lidar_points_1'] = lidar_points[up_idx]
        split_data['lidar_points_2'] = lidar_points[down_idx]

    elif X_thr is not None and Y_thr is None:
        # Horizontal 2-Split (Left/Right)
        left_idx = (img_points[:, 0] >= 0) & (img_points[:, 0] <= X_thr)
        right_idx = (img_points[:, 0] > X_thr) & (img_points[:, 0] <= width)

        split_data['img_points_1'] = img_points[left_idx]
        split_data['img_points_2'] = img_points[right_idx]
        split_data['lidar_points_1'] = lidar_points[left_idx]
        split_data['lidar_points_2'] = lidar_points[right_idx]

    elif Y_thr is not None and X_thr is not None:
        # 4-Split (Upper Left, Upper Right, Lower Left, Lower Right)
        ul_idx = (img_points[:, 0] >= 0) & (img_points[:, 0] <= X_thr) & (img_points[:, 1] >= 0) & (img_points[:, 1] <= Y_thr)
        ur_idx = (img_points[:, 0] > X_thr) & (img_points[:, 0] <= width) & (img_points[:, 1] >= 0) & (img_points[:, 1] <= Y_thr)
        ll_idx = (img_points[:, 0] >= 0) & (img_points[:, 0] <= X_thr) & (img_points[:, 1] > Y_thr) & (img_points[:, 1] <= height)
        lr_idx = (img_points[:, 0] > X_thr) & (img_points[:, 0] <= width) & (img_points[:, 1] > Y_thr) & (img_points[:, 1] <= height)

        split_data['img_points_1'] = img_points[ul_idx]
        split_data['img_points_2'] = img_points[ur_idx]
        split_data['img_points_3'] = img_points[ll_idx]
        split_data['img_points_4'] = img_points[lr_idx]
        split_data['lidar_points_1'] = lidar_points[ul_idx]
        split_data['lidar_points_2'] = lidar_points[ur_idx]
        split_data['lidar_points_3'] = lidar_points[ll_idx]
        split_data['lidar_points_4'] = lidar_points[lr_idx]

    else:
        raise ValueError("At least one of Y_thr or X_thr must be provided.")

    return split_data

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
    
    
def plot_paired_points_split2_spec(img_fr, img_points, lidar_points, i, path, matrix, matrix_1, matrix_2,Y_thr, X_thr):   
    # Get the image dimensions
    height, width = img_fr.shape[:2]
    ### Split to 2 planes to do calibration seperatly    
    transformed_points = np.squeeze(cv2.perspectiveTransform(lidar_points.reshape(-1, 1, 2), matrix))
    split_data = split_points(transformed_points, lidar_points, width, height, Y_thr, X_thr) #transformed_points as img_points
    img_points_1 = split_data['img_points_1']  
    img_points_2 = split_data['img_points_2']
    lidar_points_1 = split_data['lidar_points_1']
    lidar_points_2 = split_data['lidar_points_2']   
    
    # Apply the perspective transformation using homography_matrix
    transformed_points_1 = cv2.perspectiveTransform(lidar_points_1.reshape(-1, 1, 2), matrix_1)
    transformed_points_2 = cv2.perspectiveTransform(lidar_points_2.reshape(-1, 1, 2), matrix)
    # Initialize an empty list to store the non-empty transformed points
    transformed_points_list = []
    # Check each transformed points array and add non-empty ones to the list
    if img_points_1.size > 0:
        transformed_points_list.append(transformed_points_1)
    if img_points_2.size > 0:
        transformed_points_list.append(transformed_points_2)
    # Concatenate all non-empty transformed points
    if transformed_points_list:
        transformed_points = np.concatenate(transformed_points_list, axis=0)
    else:
        transformed_points = np.array([])  # Handle the case where all img_points are empty

        
    # Create a new figure for each plot
    plt.figure(dpi=300)
    
    # Plot the image
    plt.imshow(img_fr)
    plt.axis('off')  # Turn off the axis

    # Plot the image points
    if any(img_points):
        plt.scatter(img_points[:, 0], img_points[:, 1], s=4, c='red', label='Image')


    
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
    
def plot_paired_points_split2(img_fr, img_points, lidar_points, i, path, matrix, matrix_1, matrix_2,Y_thr, X_thr):   
    # Get the image dimensions
    height, width = img_fr.shape[:2]
    ### Split to 2 planes to do calibration seperatly    
    transformed_points =np.squeeze( cv2.perspectiveTransform(lidar_points.reshape(-1, 1, 2), matrix) )
    split_data = split_points(transformed_points, lidar_points, width, height, Y_thr, X_thr) #transformed_points as img_points
    img_points_1 = split_data['img_points_1']  
    img_points_2 = split_data['img_points_2']
    lidar_points_1 = split_data['lidar_points_1']
    lidar_points_2 = split_data['lidar_points_2']   
    
    # Apply the perspective transformation using homography_matrix
    transformed_points_1 = cv2.perspectiveTransform(lidar_points_1.reshape(-1, 1, 2), matrix_1)
    transformed_points_2 = cv2.perspectiveTransform(lidar_points_2.reshape(-1, 1, 2), matrix_2)
    # Initialize an empty list to store the non-empty transformed points
    transformed_points_list = []
    # Check each transformed points array and add non-empty ones to the list
    if img_points_1.size > 0:
        transformed_points_list.append(transformed_points_1)
    if img_points_2.size > 0:
        transformed_points_list.append(transformed_points_2)
    # Concatenate all non-empty transformed points
    if transformed_points_list:
        transformed_points = np.concatenate(transformed_points_list, axis=0)
    else:
        transformed_points = np.array([])  # Handle the case where all img_points are empty

        
    # Create a new figure for each plot
    plt.figure(dpi=300)
    
    # Plot the image
    plt.imshow(img_fr)
    plt.axis('off')  # Turn off the axis

    # Plot the image points
    if any(img_points):
        plt.scatter(img_points[:, 0], img_points[:, 1], s=4, c='red', label='Image')

    
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
    
def plot_paired_points_split4(img_fr, img_points, lidar_points, i, path, matrix, matrix_1, matrix_2, matrix_3, matrix_4,Y_thr, X_thr):   
    # Get the image dimensions
    height, width = img_fr.shape[:2]
    ### Split to 4 planes to do calibration seperatly 
    transformed_points = np.squeeze( cv2.perspectiveTransform(lidar_points.reshape(-1, 1, 2), matrix) )
    # Filter out the points that are outside the image boundaries
    #valid_indices = get_valid_point_indices(transformed_points, width, height)
    split_data = split_points(transformed_points, lidar_points, width, height, Y_thr, X_thr) #transformed_points as img_points
    img_points_1 = split_data['img_points_1']  
    img_points_2 = split_data['img_points_2']
    img_points_3 = split_data['img_points_3'] 
    img_points_4 = split_data['img_points_4']
    lidar_points_1 = split_data['lidar_points_1']
    lidar_points_2 = split_data['lidar_points_2']    
    lidar_points_3 = split_data['lidar_points_3']
    lidar_points_4 = split_data['lidar_points_4']  
    
    # Apply the perspective transformation using homography_matrix
    transformed_points_1 = cv2.perspectiveTransform(lidar_points_1.reshape(-1, 1, 2), matrix_1)
    transformed_points_2 = cv2.perspectiveTransform(lidar_points_2.reshape(-1, 1, 2), matrix_2)
    transformed_points_3 = cv2.perspectiveTransform(lidar_points_3.reshape(-1, 1, 2), matrix_3)
    transformed_points_4 = cv2.perspectiveTransform(lidar_points_4.reshape(-1, 1, 2), matrix_4)
    # Initialize an empty list to store the non-empty transformed points
    transformed_points_list = []
    # Check each transformed points array and add non-empty ones to the list
    if img_points_1.size > 0:
        transformed_points_list.append(transformed_points_1)
    if img_points_2.size > 0:
        transformed_points_list.append(transformed_points_2)
    if img_points_3.size > 0:
        transformed_points_list.append(transformed_points_3)
    if img_points_4.size > 0:
        transformed_points_list.append(transformed_points_4)
    # Concatenate all non-empty transformed points
    if transformed_points_list:
        transformed_points = np.concatenate(transformed_points_list, axis=0)
    else:
        transformed_points = np.array([])  # Handle the case where all img_points are empty

        
    # Create a new figure for each plot
    plt.figure(dpi=300)
    
    # Plot the image
    plt.imshow(img_fr)
    plt.axis('off')  # Turn off the axis

    # Plot the image points
    if any(img_points):
        plt.scatter(img_points[:, 0], img_points[:, 1], s=4, c='red', label='Image')

    
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
    img_w = 1280
    img_h = 720
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
    
    homography_matrix_21   = np.load(path+'/homography_matrix_lid2cam_21.npy', allow_pickle=True)
    homography_matrix_22   = np.load(path+'/homography_matrix_lid2cam_22.npy', allow_pickle=True)
    
    homography_matrix_1   = np.load(path+'/homography_matrix_lid2cam_1.npy', allow_pickle=True)
    homography_matrix_2   = np.load(path+'/homography_matrix_lid2cam_2.npy', allow_pickle=True)
    homography_matrix_3   = np.load(path+'/homography_matrix_lid2cam_3.npy', allow_pickle=True)
    homography_matrix_4   = np.load(path+'/homography_matrix_lid2cam_4.npy', allow_pickle=True)
    #############################  Main   #############################
    # Load the matched files data
    data = np.loadtxt(matched_files_path, delimiter=',', dtype=str)
    # # Load the data from the npy file
    # lidar_array, camera_array = extract_data_from_txt(file_path)
    # # Define camera image coordinates and radar image coordinates
    # img_points   = camera_array
    # lidar_points = lidar_array[:,:2]

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
        lidar_points = np.load(cent_file, allow_pickle=True)
        lidar_points = lidar_points[:,:2]
        img_points   = []

        ### loop end
        
        # No_split
        #plot_paired_points(img_fr, img_points, lidar_points, i, save_path, homography_matrix)
        
        # split_2
        Y_thr = img_h / 5 #img_h - (img_h / 2)
        X_thr = None #img_w - (img_w / 2)
        plot_paired_points_split2_spec(img_fr, img_points, lidar_points, i, save_path, homography_matrix, 
                                  homography_matrix_21, homography_matrix_22, Y_thr, X_thr)
        
        # # split_4
        # Y_thr = img_h / 4
        # X_thr = img_w - (img_w / 2)
        # plot_paired_points_split4(img_fr, img_points, lidar_points, i, save_path, homography_matrix, homography_matrix_1, homography_matrix_2, 
        #                           homography_matrix_3, homography_matrix_4,Y_thr, X_thr)
            
        gc.collect()
                    
                    
    # ####   END   #############################################################            
    # # Create video
    # output_filename = 'LC_paired_points.avi'
    # create_video_from_images(save_path, output_filename)
    output_filename = 'LC_paired_points.mp4'
    create_HQ_video_from_images(save_path, output_filename)
