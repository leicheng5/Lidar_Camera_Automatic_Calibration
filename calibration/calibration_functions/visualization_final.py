#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""
############### Lei ###################################
import os, glob, shutil, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc
#######################################################


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
    plt.savefig(path + filename, dpi=300)

    # Show the plot
    #plt.show()

    # Clear the current figure and remove previously plotted points
    plt.clf()

    # Close the figure
    plt.close()
  

   
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



###########################################################################
if __name__ == "__main__":
    path = r"E:\FHWA\calibration"
    output_dir = r'E:\FHWA\calibration\outputs'
    matched_files_path = os.path.join(output_dir, r'matched_files.txt')
    save_path = os.path.join(path, r'result_images/')
    input_cam_dir = r'E:\FHWA\camera_label\camera'    
    input_cents_dir = r'E:\FHWA\voxel_detector\centroids'

    
    #############################  Load calibration matrix   #############################
    homography_matrix = np.load(os.path.join(path, r'homography_matrix.npy'), allow_pickle=True)
    
    #############################  Main   #############################
    # Load the matched files data
    data = np.loadtxt(matched_files_path, delimiter=',', dtype=str)


    for i, entry in enumerate(data):
        print(f'Plot image {i}')
        # Construct full paths for each file
        img_file = os.path.join(input_cam_dir, entry[0].strip())
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
    ###########################  Paired radar and image  #############################  

 

