#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""

import cv2, os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import zscore

def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data 

def remove_outliers_zscore(data, threshold=3):
    # remove outliers using z-score and return filtered data's indices
    z_scores = np.abs(zscore(data, axis=0))
    filtered_indices = np.where((z_scores < threshold).all(axis=1))[0]
    #filtered_data = data[filtered_indices]
    return filtered_indices

def rad_cam_Homography(lidar_points, img_points,validation_lidar_points, validation_img_points, 
                       method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995):
    # Use cv2.findHomography() to calculate the perspective transformation matrix: cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    # # #RANSAC
    homography_matrix, inliers = cv2.findHomography(lidar_points, img_points, method=method, 
                                                    ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters, confidence=confidence)
    
    # #LMEDS
    # homography_matrix, inliers = cv2.findHomography(lidar_points, img_points, method=cv2.LMEDS
    #                                               , ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995)
    
    # #RHO
    # homography_matrix, inliers = cv2.findHomography(lidar_points, img_points, method=cv2.RHO
    #                                               , ransacReprojThreshold=10.0, maxIters=2000, confidence=0.995)
    
    # Output the homography matrix
    print("Homography Matrix:",homography_matrix)
    
    
    ## Outliers and Inliers
    # Apply the perspective transformation using homography_matrix
    transformed_points = cv2.perspectiveTransform(lidar_points.reshape(-1, 1, 2), homography_matrix)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points - img_points.reshape(-1, 1, 2)
    mean_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Mean Reprojection Error:", mean_reprojection_error)
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("RMSE:", rmse)
    
    
    ## Inliers
    inliers_points = lidar_points[inliers.ravel()==1] #inliers.ravel().astype(bool)
    # Get Projected Points corresponding to inliers
    transformed_inliers = transformed_points[inliers.ravel()==1]
    inliers_reprojection_errors = reprojection_errors[inliers.ravel()==1]
    mean_reprojection_error_inliers = np.mean(np.linalg.norm(inliers_reprojection_errors, axis=-1))
    # Print the inliers and their Mean Reprojection Error
    #print("Inliers:", inliers_points)
    print("Number of inliers:", len(inliers_points))
    print("Mean Reprojection Error (inliers):", mean_reprojection_error_inliers)
    # Calculate Inliers RMSE
    inliers_rmse = np.sqrt(np.mean(np.square(inliers_reprojection_errors)))
    print("Inliers RMSE:", inliers_rmse)
    
    ## Validation points
    # Apply the perspective transformation using homography_matrix
    transformed_points_val = cv2.perspectiveTransform(validation_lidar_points.reshape(-1, 1, 2), homography_matrix)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_val - validation_img_points.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Validation-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("Validation-RMSE:", val_rmse)
    
    return homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val

def rad_cam_Affine(lidar_points, img_points,validation_lidar_points, validation_img_points, 
                       method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995):
    # solve error: (-215:Assertion failed) count >= 0 && to.checkVector(2) == count
    lidar_points = lidar_points.astype(np.float32)
    img_points = img_points.astype(np.float32)

    # Use cv2.estimateAffine2D() to calculate the affine transformation matrix
    # #RANSAC
    affine_matrix, inliers = cv2.estimateAffine2D(lidar_points, img_points, method=method, 
                                                    ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters, confidence=confidence)
    
    # # #LMEDS
    # affine_matrix, inliers = cv2.estimateAffine2D(lidar_points, img_points, method=cv2.LMEDS
    #                                               , ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995)
    
    # Output the affine matrix
    print("Affine Matrix:",affine_matrix)
    
    
    ## Outliers and Inliers
    # Apply the affine transformation using affine_matrix
    transformed_points = cv2.transform(lidar_points.reshape(-1, 1, 2), affine_matrix) 
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points - img_points.reshape(-1, 1, 2)
    mean_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Mean Reprojection Error:", mean_reprojection_error)
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("RMSE:", rmse)
    
    
    ## Inliers
    inliers_points = lidar_points[inliers.ravel()==1] #inliers.ravel().astype(bool)
    # Get Projected Points corresponding to inliers
    transformed_inliers = transformed_points[inliers.ravel()==1]
    inliers_reprojection_errors = reprojection_errors[inliers.ravel()==1]
    mean_reprojection_error_inliers = np.mean(np.linalg.norm(inliers_reprojection_errors, axis=-1))
    # Print the inliers and their Mean Reprojection Error
    #print("Inliers:", inliers_points)
    print("Number of inliers:", len(inliers_points))
    print("Mean Reprojection Error (inliers):", mean_reprojection_error_inliers)
    # Calculate Inliers RMSE
    inliers_rmse = np.sqrt(np.mean(np.square(inliers_reprojection_errors)))
    print("Inliers RMSE:", inliers_rmse)
    
    ## Validation points
    # Apply the affine transformation using affine_matrix
    transformed_points_val = cv2.transform(validation_lidar_points.reshape(-1, 1, 2), affine_matrix) 
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_val - validation_img_points.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Validation-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("Validation-RMSE:", val_rmse)
    
    return affine_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val


def show_calib_result(lidar_points, img_points, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_lidar_points, transformed_points_val):
    # Plot the transformed_points and img_points with Connect the corresponding points with lines
    # Create a new figure
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    # Plot the transformed_points and img_points
    ax.scatter(img_points[:, 0], img_points[:, 1], c='blue', label='Image Points')
    ax.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='lidar Points')
    # Connect the corresponding points with lines
    for i in range(len(img_points)):
        x = [img_points[i, 0], transformed_points[i, 0, 0]]
        y = [img_points[i, 1], transformed_points[i, 0, 1]]
        ax.plot(x, y, color='green', alpha=0.5)
    # Set plot title and labels
    ax.set_title('lidar-Image Point Correspondences')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Invert the Y-axis
    ax.invert_yaxis()
    # Show legend in the upper left corner
    ax.legend(loc='upper left')
    ax.grid(True)
    # Show the plot
    plt.show()
    
    
    # Plot the transformed_points and img_points
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(img_points[:, 0], img_points[:, 1], c='blue', label='Image Points')
    plt.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='lidar Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('lidar Points vs Image Points')
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(img_points[:, 0], img_points[:, 1], c='blue', label='Image Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Image Points')
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='lidar Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('lidar Points')
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(lidar_points[:, 0], lidar_points[:, 1], c='red', label='lidar Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original lidar Points')
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    # Plot the transformed_inliers and inliers_points for inliers
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(inliers_points[:, 0], inliers_points[:, 1], c='blue', label='Original lidar Points')
    plt.scatter(transformed_inliers[:, 0, 0], transformed_inliers[:, 0, 1], c='red', label='lidar Points')
    # Connect the corresponding points with lines
    for i in range(len(transformed_inliers)):
        x = [inliers_points[i, 0], transformed_inliers[i, 0, 0]]
        y = [inliers_points[i, 1], transformed_inliers[i, 0, 1]]
        plt.plot(x, y, color='green', alpha=0.5)
    # Set plot title and labels
    plt.title('lidar-Original lidar Inlier Correspondences')
    plt.xlabel('X')
    plt.ylabel('Y')
    # Show legend and grid
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    # Show the plot
    plt.show()
    
    
    # Plot the transformed_inliers and inliers_points for inliers
    plt.figure(figsize=(8, 6), dpi=300)
    img_inliers = img_points[inliers.ravel() == 1, :]
    plt.scatter(img_inliers[:, 0], img_inliers[:, 1], c='blue', label='Image Points')
    plt.scatter(transformed_inliers[:, 0, 0], transformed_inliers[:, 0, 1], c='red', label='lidar Points')
    # Connect the corresponding points with lines
    for i in range(len(transformed_inliers)):
        x = [img_inliers[i, 0], transformed_inliers[i, 0, 0]]
        y = [img_inliers[i, 1], transformed_inliers[i, 0, 1]]
        plt.plot(x, y, color='green', alpha=0.5)
    # Set plot title and labels
    plt.title('lidar-Image Inlier Correspondences')
    plt.xlabel('X')
    plt.ylabel('Y')
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    # Show the plot
    plt.show()
    
    
    # Plot the transformed_points and img_points with Connect the corresponding points with lines
    # Create a new figure
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    # Plot the transformed_points and img_points
    ax.scatter(img_points[:, 0], img_points[:, 1], c='blue', label='Image Points')
    ax.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='lidar Points')
    # Connect the corresponding points with lines
    for i in range(len(img_points)):
        x = [img_points[i, 0], transformed_points[i, 0, 0]]
        y = [img_points[i, 1], transformed_points[i, 0, 1]]
        ax.plot(x, y, color='green', alpha=0.5)
    
    # Mark the inliers with hollow circles
    ax.scatter(img_points[inliers.ravel() == 1, 0], img_points[inliers.ravel() == 1, 1], c='none', label='Inliers', facecolors='none', edgecolors='cyan', marker='o', s=50)
    ax.scatter(transformed_points[inliers.ravel() == 1, 0, 0], transformed_points[inliers.ravel() == 1, 0, 1], c='none', facecolors='none', edgecolors='cyan', marker='o', s=50) #marker='^'
    
    # Set plot title and labels
    ax.set_title('lidar-Image Point Correspondences')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Invert the Y-axis
    ax.invert_yaxis()
    # Show legend in the upper left corner
    ax.legend(loc='upper left')
    ax.grid(True)
    # Show the plot
    plt.show()
	
    # Plot the transformed_points_val and validation_img_points with Connect the corresponding points with lines
    # Create a new figure
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    # Plot the transformed_points_val and validation_img_points
    ax.scatter(validation_img_points[:, 0], validation_img_points[:, 1], c='blue', label='Image Points')
    ax.scatter(transformed_points_val[:, 0, 0], transformed_points_val[:, 0, 1], c='red', label='lidar Points')
    # Connect the corresponding points with lines
    for i in range(len(validation_img_points)):
        x = [validation_img_points[i, 0], transformed_points_val[i, 0, 0]]
        y = [validation_img_points[i, 1], transformed_points_val[i, 0, 1]]
        ax.plot(x, y, color='green', alpha=0.5)
    
    # Set plot title and labels
    ax.set_title('Validation lidar-Image Point Correspondences')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Invert the Y-axis
    ax.invert_yaxis()
    # Show legend in the upper left corner
    ax.legend(loc='upper left')
    ax.grid(True)
    # Show the plot
    plt.show()	
    


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


def Split_2_Validation(homography_matrix_up, homography_matrix_down, validation_lidar_points, validation_img_points, img_w, img_h,Y_thr=None,X_thr=None):
    ### Split to multiple planes to do calibration seperatly
    # Split the points based on the Y-coordinate threshold
    #Y_thr = 320  # img_h - (img_h / 3)
    # up_idx = validation_img_points[:, XorY] <= Y_thr
    # down_idx = validation_img_points[:, XorY] > Y_thr
    split_data = split_points(validation_img_points, validation_lidar_points, img_w, img_h, Y_thr, X_thr)
    img_points_up = split_data['img_points_1']  # same with left
    img_points_down = split_data['img_points_2']
    lidar_points_up = split_data['lidar_points_1']
    lidar_points_down = split_data['lidar_points_2']    
    
    ### Validation points UP
    # Apply the perspective transformation using homography_matrix
    transformed_points_up = cv2.perspectiveTransform(lidar_points_up.reshape(-1, 1, 2), homography_matrix_up)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_up - img_points_up.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("UPorLeft-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("UPorLeft-RMSE:", val_rmse)
    
    ### Validation points DOWN
    # Apply the perspective transformation using homography_matrix
    transformed_points_down = cv2.perspectiveTransform(lidar_points_down.reshape(-1, 1, 2), homography_matrix_down)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_down - img_points_down.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("DOWNorRight-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("DOWNorRight-RMSE:", val_rmse)
    
    # Plot the transformed_points_val and validation_img_points with Connect the corresponding points with lines
    # Create a new figure
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    # Plot the transformed_points_up and validation_img_points
    ax.scatter(img_points_up[:, 0], img_points_up[:, 1], c='blue', label='Image Points')
    ax.scatter(transformed_points_up[:, 0, 0], transformed_points_up[:, 0, 1], c='red', label='lidar Points')
    # Connect the corresponding points with lines
    for i in range(len(img_points_up)):
        x = [img_points_up[i, 0], transformed_points_up[i, 0, 0]]
        y = [img_points_up[i, 1], transformed_points_up[i, 0, 1]]
        ax.plot(x, y, color='green', alpha=0.5)
        
    # Plot the transformed_points_down and validation_img_points
    ax.scatter(img_points_down[:, 0], img_points_down[:, 1], c='blue')
    ax.scatter(transformed_points_down[:, 0, 0], transformed_points_down[:, 0, 1], c='red')
    # Connect the corresponding points with lines
    for i in range(len(img_points_down)):
        x = [img_points_down[i, 0], transformed_points_down[i, 0, 0]]
        y = [img_points_down[i, 1], transformed_points_down[i, 0, 1]]
        ax.plot(x, y, color='orange', alpha=0.5)        
    
    # Set plot title and labels
    ax.set_title('Validation lidar-Image Point Correspondences')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Invert the Y-axis
    ax.invert_yaxis()
    # Show legend in the upper left corner
    ax.legend(loc='upper left')
    ax.grid(True)
    # Show the plot
    plt.show()	    
    
    
def Split_4_Validation(homography_matrix_1, homography_matrix_2, homography_matrix_3, homography_matrix_4, validation_lidar_points, validation_img_points, img_w, img_h,Y_thr=None,X_thr=None):
    ### Split to multiple planes to do calibration seperatly (Upper Left, Upper Right, Lower Left, Lower Right)
    split_data = split_points(validation_img_points, validation_lidar_points, img_w, img_h, Y_thr, X_thr)
    img_points_1 = split_data['img_points_1']  
    img_points_2 = split_data['img_points_2']
    img_points_3 = split_data['img_points_3'] 
    img_points_4 = split_data['img_points_4']
    lidar_points_1 = split_data['lidar_points_1']
    lidar_points_2 = split_data['lidar_points_2']    
    lidar_points_3 = split_data['lidar_points_3']
    lidar_points_4 = split_data['lidar_points_4']
    
    ### Validation points Upper_Left
    # Apply the perspective transformation using homography_matrix
    transformed_points_1 = cv2.perspectiveTransform(lidar_points_1.reshape(-1, 1, 2), homography_matrix_1)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_1 - img_points_1.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Upper_Left-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("Upper_Left-RMSE:", val_rmse)
    
    ### Validation points Upper_Right, Lower Left, Lower Right)
    # Apply the perspective transformation using homography_matrix
    transformed_points_2 = cv2.perspectiveTransform(lidar_points_2.reshape(-1, 1, 2), homography_matrix_2)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_2 - img_points_2.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Upper_Right-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("Upper_Right-RMSE:", val_rmse)
    
    ### Validation points Lower_Left
    # Apply the perspective transformation using homography_matrix
    transformed_points_3 = cv2.perspectiveTransform(lidar_points_3.reshape(-1, 1, 2), homography_matrix_3)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_3 - img_points_3.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Lower_Left-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("Lower_Left-RMSE:", val_rmse)
    
    ### Validation points Lower_Right
    # Apply the perspective transformation using homography_matrix
    transformed_points_4 = cv2.perspectiveTransform(lidar_points_4.reshape(-1, 1, 2), homography_matrix_4)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_4 - img_points_4.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Lower_Right-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("Lower_Right-RMSE:", val_rmse)
    
    # Plot the Upper_Left transformed_points_val and validation_img_points with Connect the corresponding points with lines
    # Create a new figure
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    # Plot the transformed_points_up and validation_img_points
    ax.scatter(img_points_1[:, 0], img_points_1[:, 1], c='blue', label='Image Points')
    ax.scatter(transformed_points_1[:, 0, 0], transformed_points_1[:, 0, 1], c='red', label='lidar Points')
    # Connect the corresponding points with lines
    for i in range(len(img_points_1)):
        x = [img_points_1[i, 0], transformed_points_1[i, 0, 0]]
        y = [img_points_1[i, 1], transformed_points_1[i, 0, 1]]
        ax.plot(x, y, color='green', alpha=0.5)
        
    # Plot the Upper_Right transformed_points_down and validation_img_points
    ax.scatter(img_points_2[:, 0], img_points_2[:, 1], c='blue')
    ax.scatter(transformed_points_2[:, 0, 0], transformed_points_2[:, 0, 1], c='red')
    # Connect the corresponding points with lines
    for i in range(len(img_points_2)):
        x = [img_points_2[i, 0], transformed_points_2[i, 0, 0]]
        y = [img_points_2[i, 1], transformed_points_2[i, 0, 1]]
        ax.plot(x, y, color='orange', alpha=0.5)    

    # Plot the Lower_Left transformed_points_down and validation_img_points
    ax.scatter(img_points_3[:, 0], img_points_3[:, 1], c='blue')
    ax.scatter(transformed_points_3[:, 0, 0], transformed_points_3[:, 0, 1], c='red')
    # Connect the corresponding points with lines
    for i in range(len(img_points_3)):
        x = [img_points_3[i, 0], transformed_points_3[i, 0, 0]]
        y = [img_points_3[i, 1], transformed_points_3[i, 0, 1]]
        ax.plot(x, y, color='purple', alpha=0.5)  

    # Plot the Lower_Right transformed_points_down and validation_img_points
    ax.scatter(img_points_4[:, 0], img_points_4[:, 1], c='blue')
    ax.scatter(transformed_points_4[:, 0, 0], transformed_points_4[:, 0, 1], c='red')
    # Connect the corresponding points with lines
    for i in range(len(img_points_4)):
        x = [img_points_4[i, 0], transformed_points_4[i, 0, 0]]
        y = [img_points_4[i, 1], transformed_points_4[i, 0, 1]]
        ax.plot(x, y, color='cyan', alpha=0.5)          
    
    
    # Set plot title and labels
    ax.set_title('Validation lidar-Image Point Correspondences')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Invert the Y-axis
    ax.invert_yaxis()
    # Show legend in the upper left corner
    ax.legend(loc='upper left')
    ax.grid(True)
    # Show the plot
    plt.show()	    
#####################################################################################

if __name__ == "__main__":
    ###########################  Load Centers  #########################################
    path = './calib_results'
    img_w = 1440
    img_h = 1080
    
    ####################################################################################
    # Define additional Validation points
    filename = os.path.join(path,'selected_pairs_array_confid_valid.npy')
    # Load the data from the npy file
    data = np.load(filename, allow_pickle=True)
    data = np.array(data)
    # Define camera image coordinates and lidar image coordinates
    validation_img_points   = data[:,0]
    validation_lidar_points = data[:,1]
    
    
    ##########################  Points used for Calibration    ##########################################################
    filename = os.path.join(path,'selected_pairs_array_confid.npy')
    # Load the data from the npy file
    data = np.load(filename, allow_pickle=True)
    # Define camera image coordinates and lidar image coordinates
    img_points   = data[:,0]
    lidar_points = data[:,1]
    # # Remove outliers from the data points
    # filtered_indices = remove_outliers_zscore(img_points, threshold=3)
    # img_points       = img_points[filtered_indices]
    # lidar_points     = lidar_points[filtered_indices]
    ####################################################################################
    
    
    USE_findHomography   = 1
    USE_estimateAffine2D = 0
    if USE_findHomography:
        ### For all points
        homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
            rad_cam_Homography(lidar_points, img_points, validation_lidar_points, validation_img_points, 
                              method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
        # Save  as an .npy file
        np.save(path+'/homography_matrix_lid2cam.npy', homography_matrix) 
        # Plot calibration results
        show_calib_result(lidar_points, img_points, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_lidar_points, transformed_points_val)
        

        mode = None #'split_2'
        if mode == 'split_2':
            Y_thr = img_h / 3 #img_h - (img_h / 2)
            X_thr = None #img_w - (img_w / 2)
            ### Split to 2 planes to do calibration seperatly 
            split_data = split_points(img_points, lidar_points, img_w, img_h, Y_thr, X_thr)
            img_points_1 = split_data['img_points_1']  
            img_points_2 = split_data['img_points_2']
            lidar_points_1 = split_data['lidar_points_1']
            lidar_points_2 = split_data['lidar_points_2']    
            ### For UP
            homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
                rad_cam_Homography(lidar_points_1, img_points_1, validation_lidar_points, validation_img_points, 
                                  method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
            # Save  as an .npy file
            np.save(path+'/homography_matrix_lid2cam_21.npy', homography_matrix) 
            # Plot calibration results
            show_calib_result(lidar_points_1, img_points_1, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_lidar_points, transformed_points_val)
            
            ### For DOWN
            homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
                rad_cam_Homography(lidar_points_2, img_points_2, validation_lidar_points, validation_img_points, 
                                  method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
            # Save  as an .npy file
            np.save(path+'/homography_matrix_lid2cam_22.npy', homography_matrix) 
            # Plot calibration results
            show_calib_result(lidar_points_2, img_points_2, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_lidar_points, transformed_points_val)    
    
            ### Validation For UP and DOWN
            homography_matrix_1   = np.load(path+'/homography_matrix_lid2cam_21.npy', allow_pickle=True)
            homography_matrix_2   = np.load(path+'/homography_matrix_lid2cam_22.npy', allow_pickle=True)
            Split_2_Validation(homography_matrix_1, homography_matrix_2, validation_lidar_points, validation_img_points, img_w, img_h,Y_thr,X_thr)   
        elif mode == 'split_4':   
            Y_thr = img_h / 4
            X_thr = img_w - (img_w / 2)
            ### Split to multiple planes to do calibration seperatly (Upper Left, Upper Right, Lower Left, Lower Right)
            split_data = split_points(img_points, lidar_points, img_w, img_h, Y_thr, X_thr)
            img_points_1 = split_data['img_points_1']  
            img_points_2 = split_data['img_points_2']
            img_points_3 = split_data['img_points_3'] 
            img_points_4 = split_data['img_points_4']
            lidar_points_1 = split_data['lidar_points_1']
            lidar_points_2 = split_data['lidar_points_2']    
            lidar_points_3 = split_data['lidar_points_3']
            lidar_points_4 = split_data['lidar_points_4']
            ### For Upper Left
            homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
                rad_cam_Homography(lidar_points_1, img_points_1, validation_lidar_points, validation_img_points, 
                                  method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
            # Save  as an .npy file
            np.save(path+'/homography_matrix_lid2cam_1.npy', homography_matrix) 
            # Plot calibration results
            show_calib_result(lidar_points_1, img_points_1, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_lidar_points, transformed_points_val)
            
            ### For Upper Right
            homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
                rad_cam_Homography(lidar_points_2, img_points_2, validation_lidar_points, validation_img_points, 
                                  method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
            # Save  as an .npy file
            np.save(path+'/homography_matrix_lid2cam_2.npy', homography_matrix) 
            # Plot calibration results
            show_calib_result(lidar_points_2, img_points_2, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_lidar_points, transformed_points_val)    
            
            ### For Lower Left
            homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
                rad_cam_Homography(lidar_points_3, img_points_3, validation_lidar_points, validation_img_points, 
                                  method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
            # Save  as an .npy file
            np.save(path+'/homography_matrix_lid2cam_3.npy', homography_matrix) 
            # Plot calibration results
            show_calib_result(lidar_points_3, img_points_3, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_lidar_points, transformed_points_val)
            
            
            ### For Lower Right
            homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
                rad_cam_Homography(lidar_points_4, img_points_4, validation_lidar_points, validation_img_points, 
                                  method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
            # Save  as an .npy file
            np.save(path+'/homography_matrix_lid2cam_4.npy', homography_matrix) 
            # Plot calibration results
            show_calib_result(lidar_points_4, img_points_4, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_lidar_points, transformed_points_val)    
    
            ### Validation For 4 planes
            homography_matrix_1   = np.load(path+'/homography_matrix_lid2cam_1.npy', allow_pickle=True)
            homography_matrix_2   = np.load(path+'/homography_matrix_lid2cam_2.npy', allow_pickle=True)
            homography_matrix_3   = np.load(path+'/homography_matrix_lid2cam_3.npy', allow_pickle=True)
            homography_matrix_4   = np.load(path+'/homography_matrix_lid2cam_4.npy', allow_pickle=True)
            Split_4_Validation(homography_matrix_1, homography_matrix_2,homography_matrix_3, homography_matrix_4, validation_lidar_points, validation_img_points, img_w, img_h,Y_thr,X_thr)               


    
    # Output the Projected Points
    #print("Projected Points:",transformed_points)


