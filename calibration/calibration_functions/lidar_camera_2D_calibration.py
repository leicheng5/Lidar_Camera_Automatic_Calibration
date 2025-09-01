#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""

import os, cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_data_from_txt(file_path):
    # Load the data from the file
    data = np.loadtxt(file_path, delimiter=',')
    
    # Split the data into LiDAR and camera arrays
    lidar_array = data[:, :3]
    camera_array = data[:, 3:5]
    
    return lidar_array, camera_array
###########################  Load Centers  #########################################
path = r"E:\FHWA\calibration"
file_path = os.path.join(path, r'picking_list_all.txt')
# Load the data from the npy file
lidar_array, camera_array = extract_data_from_txt(file_path)
# Define camera image coordinates and radar image coordinates
img_points   = camera_array
radar_points = lidar_array[:,:2]
####################################################################################
# Define additional validation points
validation_img_points = img_points
validation_radar_points = radar_points

USE_findHomography   = 1
if USE_findHomography:
    # Use cv2.findHomography() to calculate the perspective transformation matrix: cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    # #RANSAC
    # homography_matrix, inliers = cv2.findHomography(radar_points, img_points, method=cv2.RANSAC
    #                                               , ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995)
    
    # #LMEDS
    homography_matrix, inliers = cv2.findHomography(radar_points, img_points, method=cv2.LMEDS
                                                  , ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995)
    
    # #RHO
    # homography_matrix, inliers = cv2.findHomography(radar_points, img_points, method=cv2.RHO
    #                                               , ransacReprojThreshold=10.0, maxIters=2000, confidence=0.995)
    
    # Output the homography matrix
    print("Homography Matrix:",homography_matrix)    

    # Apply the perspective transformation using homography_matrix
    transformed_points = cv2.perspectiveTransform(validation_radar_points.reshape(-1, 1, 2), homography_matrix)
    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points - validation_img_points.reshape(-1, 1, 2)
    mean_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Mean Reprojection Error:", mean_reprojection_error)
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("RMSE:", rmse)
    
    ## Inliers
    inliers_points = radar_points[inliers.ravel()==1] #inliers.ravel().astype(bool)
    # Get Projected Points corresponding to inliers
    transformed_inliers = transformed_points[inliers.ravel()==1]
    inliers_reprojection_errors = reprojection_errors[inliers.ravel()==1]
    mean_reprojection_error_inliers = np.mean(np.linalg.norm(inliers_reprojection_errors, axis=-1))
    # Print the inliers and their Mean Reprojection Error
    print("Inliers:", inliers_points, "\nNumber of inliers:", len(inliers_points))
    print("Mean Reprojection Error (inliers):", mean_reprojection_error_inliers)
    # Calculate Inliers RMSE
    inliers_rmse = np.sqrt(np.mean(np.square(inliers_reprojection_errors)))
    print("Inliers RMSE:", inliers_rmse)
    
    # Save  as an .npy file
    np.save(os.path.join(path, r'homography_matrix.npy'), homography_matrix) 
    



# Output the Projected Points
#print("Projected Points:",transformed_points)


# Plot the transformed_points and validation_img_points with Connect the corresponding points with lines
# Create a new figure
fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111)
# Plot the transformed_points and validation_img_points
ax.scatter(validation_img_points[:, 0], validation_img_points[:, 1], c='blue', label='Image Points')
ax.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='Lidar Points')
# Connect the corresponding points with lines
for i in range(len(validation_img_points)):
    x = [validation_img_points[i, 0], transformed_points[i, 0, 0]]
    y = [validation_img_points[i, 1], transformed_points[i, 0, 1]]
    ax.plot(x, y, color='green', alpha=0.5)
# Set plot title and labels
ax.set_title('Correspondence between Lidar Points and Image Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
# Show legend and grid
ax.legend()
ax.grid(True)
# Show the plot
plt.show()


# Plot the transformed_points and validation_img_points
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(validation_img_points[:, 0], validation_img_points[:, 1], c='blue', label='Image Points')
plt.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='Lidar Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Lidar Points vs Image Points')
plt.legend()
plt.grid(True)
plt.show()

# Plot the transformed_inliers and inliers_points for inliers
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(inliers_points[:, 0], inliers_points[:, 1], c='blue', label='Inliers Points')
plt.scatter(transformed_inliers[:, 0, 0], transformed_inliers[:, 0, 1], c='red', label='Lidar Points')
# Connect the corresponding points with lines
for i in range(len(transformed_inliers)):
    x = [inliers_points[i, 0], transformed_inliers[i, 0, 0]]
    y = [inliers_points[i, 1], transformed_inliers[i, 0, 1]]
    plt.plot(x, y, color='green', alpha=0.5)
# Set plot title and labels
plt.title('Correspondence between Transformed Inliers and Inliers Points')
plt.xlabel('X')
plt.ylabel('Y')
# Show legend and grid
plt.legend()
plt.grid(True)
# Show the plot
plt.show()

# Plot the transformed_points and validation_img_points with Connect the corresponding points with lines
# Create a new figure
fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111)
# Plot the transformed_points and validation_img_points
ax.scatter(validation_img_points[:, 0], validation_img_points[:, 1], c='blue', label='Image Points')
ax.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='Lidar Points')
# Connect the corresponding points with lines
for i in range(len(validation_img_points)):
    x = [validation_img_points[i, 0], transformed_points[i, 0, 0]]
    y = [validation_img_points[i, 1], transformed_points[i, 0, 1]]
    ax.plot(x, y, color='green', alpha=0.5)

# Mark the inliers with hollow circles
ax.scatter(validation_img_points[inliers.ravel() == 1, 0], validation_img_points[inliers.ravel() == 1, 1], c='none', label='Inliers', facecolors='none', edgecolors='cyan', marker='o', s=50)
ax.scatter(transformed_points[inliers.ravel() == 1, 0, 0], transformed_points[inliers.ravel() == 1, 0, 1], c='none', facecolors='none', edgecolors='cyan', marker='o', s=50) #marker='^'

# Set plot title and labels
ax.set_title('Correspondence between Lidar Points and Image Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
# Show legend and grid
ax.legend()
ax.grid(True)
# Show the plot
plt.show()