#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:34:51 2025

@author: leicheng
"""

import cv2
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image, ImageDraw

SHOW = False

def save_data_dict(data_dict, filename, directory):
    """
    Save the data_dict as a pickle file to the specified directory with the filename.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Create the full path for the pickle file
    file_path = os.path.join(directory, filename)

    # Save the data_dict to a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)
        
        
def plot_point_cloud_3d(point_cloud):
    """
    Plots a 3D scatter plot of the point cloud data.

    Parameters:
    point_cloud (numpy array): A NumPy array of shape (N, 3) where N is the number of points,
                               and the columns represent x, y, z coordinates.
    """
    # Extract x, y, and z coordinates
    x_coords = point_cloud[:, 0]
    y_coords = point_cloud[:, 1]
    z_coords = point_cloud[:, 2]
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, s=1, c='blue', marker='o')  # s is the marker size, c is the color
    # # Add labels and title
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # ax.set_zlabel('Z Coordinate')
    # ax.set_title('3D Point Cloud Scatter Plot')
    # Show the plot
    plt.show()

def annotate_bboxes_on_image(image, detections, outline="red", width=3, show=False, save_path=None):
    """
    Draw all bounding boxes from `detections` onto the PIL image.

    Parameters:
        image      : PIL.Image to draw on (will be modified in place).
        detections : List of dicts, each with det['bbox'] = {'x1','y1','x2','y2'}.
        outline    : Color for the rectangle outline (default: "red").
        width      : Stroke width in pixels (default: 3).
        show       : If True, call image.show() after drawing.
        save_path  : If not None, save the annotated image to this path.

    Returns:
        The annotated PIL.Image.
    """
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1 = det['bbox']['x1']
        y1 = det['bbox']['y1']
        x2 = det['bbox']['x2']
        y2 = det['bbox']['y2']
        # Draw the rectangle
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)

    if show:
        image.show()
    if save_path:
        image.save(save_path)
    return image
    
def show_image(img):
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()    
     

            

#### Rectify image
# def rectify_image(image, K, dist):
#     """
#     Rectify (undistort) the input PIL image using the given camera matrix and distortion coefficients.
#     """
#     # Convert PIL Image (RGB) to OpenCV BGR format
#     img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#     # Undistort image
#     # nk = K.copy()
#     # scale = 2
#     # nk[0,0]=K[0,0]/scale #scaling
#     # nk[1,1]=K[1,1]/scale
#     undistorted_bgr = cv2.undistort(img_bgr, K, dist) #, None, nk
#     # Convert back to RGB and PIL Image
#     undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(undistorted_rgb)


# def rectify_image(image, K, dist, alpha=1.0):
#     """
#     Rectify a PIL image using the given camera matrix and distortion coefficients.
#     Keeps full field-of-view (no cropping) and outputs an image of same size.

#     Parameters:
#         image: PIL.Image (RGB)
#         K:     3x3 intrinsic matrix
#         dist:  distortion coefficients (k1, k2, p1, p2, k3)
#         alpha: 0=minimum black edges (tight crop), 1=keep all pixels (default)

#     Returns:
#         rect_img: undistorted PIL.Image (same size as input)
#         newK:     new optimized camera matrix
#     """
#     # Convert PIL to OpenCV BGR
#     img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#     h, w = img_bgr.shape[:2]

#     # Get optimal new camera matrix, same size, full FOV
#     newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))

#     # Identity rectification (no stereo)
#     map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)

#     # Remap image
#     undistorted_bgr = cv2.remap(img_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)

#     # Convert back to RGB and PIL
#     undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
#     rect_img = Image.fromarray(undistorted_rgb)

#     return rect_img


# def rectify_image(image, K, dist, x_shift=240, y_shift=60, alpha=1.0):
#     """
#     Rectify (undistort) a PIL image using the given camera matrix and distortion coefficients.
#     Output image is enlarged by x_shift and y_shift on each side, retaining full field-of-view.

#     Parameters:
#         image   : PIL.Image (RGB)
#         K       : 3x3 intrinsic matrix
#         dist    : distortion coefficients (k1, k2, p1, p2, k3)
#         x_shift : pixels added to left/right
#         y_shift : pixels added to top/bottom
#         alpha   : free scaling parameter (1.0 = keep all pixels)

#     Returns:
#         rect_img: undistorted PIL.Image of size (h+2*y_shift, w+2*x_shift)
#     """
#     # Convert PIL to OpenCV BGR
#     img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#     h, w = img_bgr.shape[:2]

#     # New output size
#     new_w = w + 2 * x_shift
#     new_h = h + 2 * y_shift

#     # Compute optimal new camera matrix for enlarged image
#     newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (new_w, new_h))

#     # Shift principal point to center the original image
#     newK[0, 2] += x_shift
#     newK[1, 2] += y_shift

#     # Create undistortion map
#     map1, map2 = cv2.initUndistortRectifyMap(
#         K, dist, None, newK, (new_w, new_h), cv2.CV_32FC1
#     )

#     # Remap to undistort (and enlarge)
#     undistorted_bgr = cv2.remap(img_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)

#     # Convert back to PIL RGB
#     undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
#     rect_img = Image.fromarray(undistorted_rgb)
#     return rect_img


def rectify_image(img, K, dist, alpha=0.3): #alpha=0.3
    """
    Undistort an image and return both the full undistorted output and the cropped result.

    Parameters:
        img    : Input image as a NumPy array (BGR).
        K      : Camera intrinsic matrix.
        dist   : Distortion coefficients.
        alpha  : Free scaling parameter (0 = crop tight, 1 = keep all pixels).

    Returns:
        dst_full : Undistorted image before cropping.
        dst_crop : Cropped undistorted image (valid ROI).
        newK     : New camera matrix from getOptimalNewCameraMatrix.
        roi      : Tuple (x, y, w, h) defining the valid region in dst_full.
    """

    # Convert PIL to OpenCV BGR
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # Compute optimal new camera matrix and valid ROI
    #newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w-192, h-108)) #1/10-->alpha=0.3
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))

    # Undistort the full image
    dst_full = cv2.undistort(img_bgr, K, dist, None, newK)

    # Crop the valid region from dst_full
    x, y, w_roi, h_roi = roi
    dst_crop = dst_full[y:y+h_roi, x:x+w_roi]
    
    # Convert back to PIL RGB
    dst_full = cv2.cvtColor(dst_full, cv2.COLOR_BGR2RGB)
    dst_full = Image.fromarray(dst_full)
    
    dst_crop = cv2.cvtColor(dst_crop, cv2.COLOR_BGR2RGB)
    dst_crop = Image.fromarray(dst_crop)

    #return dst_full, dst_crop, newK, roi
    return dst_full, newK


def map_point_to_rectified(pt, K, dist, newK):
    """
    Map a point (u, v) from the distorted image into the undistorted image,
    using the optimal new camera matrix newK.
    """
    # undistortPoints with P=newK will give you pixel coords in the rectified image
    pts      = np.array([[pt]], dtype=np.float32)      # shape = (1,1,2)
    und      = cv2.undistortPoints(pts, K, dist, P=newK)
    u_rect, v_rect = und[0,0]
    return float(u_rect), float(v_rect)

def rectify_bbox_dict(bbox, K, dist, newK):
    """
    Rectify a single bbox defined by 'x1','y1','x2','y2'.
    Returns a new dict in the same format, in the undistorted image frame.
    """
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    
    # Warp each corner
    mapped = [ map_point_to_rectified(c, K, dist, newK) for c in corners ]
    us, vs = zip(*mapped)
    
    return {
        'x1': min(us),
        'y1': min(vs),
        'x2': max(us),
        'y2': max(vs)
    }

def rectify_bboxes_list(detections, K, dist, newK):
    """
    Batch-rectify all detections in a list, returning a new list
    with each 'bbox' warped into the undistorted frame.
    """
    out = []
    for det in detections:
        det2 = det.copy()
        det2['bbox'] = rectify_bbox_dict(det['bbox'], K, dist, newK)
        out.append(det2)
    return out

def gen_camera_data(data, save_dir, K, dist):
    """
    Rectify images and bboxes for each entry in camera_data,
    and save the result as one .pkl file in save_dir.
    
    Parameters:
        data: dict mapping keys to {'pixels': PIL.Image, 'detections': [...]}
        save_dir: directory where .pkl file will be saved
        K: camera intrinsic matrix
        dist: distortion coefficients
    """
    os.makedirs(save_dir, exist_ok=True)
    annotated_dir = os.path.join(save_dir,'All_rectify_imgs')
    os.makedirs(annotated_dir, exist_ok=True)

    rectified_data = {}
    img_idx = 1

    for key in sorted(data.keys()):
        entry = data[key]
        img = entry['pixels']
        dets = entry.get('detections', [])
        
        # Resize a PIL image to a target size=(1280, 720)
        #img = img.resize((1280, 720), Image.BILINEAR)

        # Rectify the image
        #rect_img = rectify_image(img, K, dist)
        rect_img, newK  = rectify_image(img, K, dist)
        # show_image(rect_img)
        #print(newK) --> newK never change

        # Rectify all bounding boxes
        rect_dets = rectify_bboxes_list(dets, K, dist, newK)
        
        # Plot all bounding boxes
        annotated_img = annotate_bboxes_on_image(rect_img,rect_dets)
        #show_image(annotated_img)
        annotated_path = os.path.join(annotated_dir, f"{img_idx}.png")
        annotated_img.save(annotated_path)
        img_idx += 1

        # Store rectified entry
        rectified_data[key] = {
            'pixels': rect_img,
            'detections': rect_dets
        }

    # # Save the entire rectified_data dict as one pickle file
    out_path = os.path.join(save_dir, "Camera_data_rectified.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(rectified_data, f)
    
    print(f"Saved rectified camera data for all keys to: {out_path}")
    return newK

def gen_label_data(data, save_dir, K, dist, newK):
    """
    Rectify camera bboxes in labeled_data and save all entries as one list in a .pkl file.
    
    Parameters:
        data: dict mapping keys to list of tuples:
              (cam_key, lid_key, cam_det, lid_det)
        save_dir: directory to save the .pkl file
        K: camera intrinsic matrix
        dist: distortion coefficients
        newK: optimized new camera matrix
    """
    os.makedirs(save_dir, exist_ok=True)
    rectified_list = []

    for key in sorted(data.keys()):
        for cam_key, lid_key, cam_det, lid_det in data[key]:
            # Copy and rectify the camera bbox
            cam_det_rect = cam_det.copy()
            cam_det_rect['bbox'] = rectify_bbox_dict(cam_det['bbox'], K, dist, newK)
            # Append the rectified tuple
            rectified_list.append((cam_key, lid_key, cam_det_rect, lid_det))

    # Save the entire list of rectified entries as one pickle file
    out_path = os.path.join(save_dir, "label_data_rectified_list.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(rectified_list, f)

    print(f"Saved {len(rectified_list)} rectified entries to: {out_path}")



def scale_intrinsic_matrix(K, from_size, to_size):
    """
    Scale intrinsic matrix K from from_size (w, h) to to_size (w, h).
    """
    scale_x = to_size[0] / from_size[0]
    scale_y = to_size[1] / from_size[1]

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled


def resize_image(img, target_size=(1280, 720)):
    """
    Resize a PIL image to a target size.

    Parameters:
        img         : PIL.Image in RGB
        target_size : (width, height) to resize to (default: 1280×720)

    Returns:
        resized_img : PIL.Image resized to target_size
    """
    resized_img = img.resize(target_size, Image.BILINEAR)
    return resized_img


    
if __name__ == '__main__':    
    # Define the base directory containing the Labeled_Data
    root     = r'/xdisk/caos/leicheng/FHWA'
    base_dir = r'/xdisk/caos/leicheng/FHWA/dataset4' #r'/xdisk/caos/leicheng/FHWA/dataset3'
    save_dir = r'/xdisk/caos/leicheng/FHWA/Lei_dataset4' #r'/xdisk/caos/leicheng/FHWA/Lei_dataset3' 
    
    
    # Camera intrinsic matrix
    # 1280 for dataset3
    K = np.array([
        [696.43755463,   0.          , 642.63789053],
        [  0.          , 695.78717319, 394.51924698],
        [  0.          ,   0.        ,   1.        ]
    ], dtype=np.float32)
    # K = np.array([
    #     [1.03975762e+03, 0.00000000e+00, 9.59541677e+02],
    #     [0.00000000e+00, 1.04074533e+03, 5.74106358e+02],
    #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    # ], dtype=np.float32)

    
    K = scale_intrinsic_matrix(K, from_size=(1280, 720), to_size=(1920, 1080))
    
    # Distortion coefficients (k1, k2, p1, p2, k3)
    # 1280 for dataset3
    dist = np.array([
        -3.79946698e-01,
          1.64050708e-01,
        -1.52008351e-03,
        -1.26369441e-04,
        -3.45314007e-02
    ], dtype=np.float32)
    # dist = np.array([
    #     -0.42524474,
    #       0.34149627,
    #     -0.00047064,
    #       0.00042540,
    #     -0.25351759
    # ], dtype=np.float32)    

    
    #dist = np.zeros((5,), dtype=np.float32)
    
    # Iterate over each subdirectory
    for folder_name in sorted(os.listdir(base_dir)):
        print('folder_name: ', folder_name)
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # Iterate over each .pkl file in the subdirectory
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(folder_path, file_name)
                    # Assign data to variables based on specific strings in the file name
                    if 'Camera' in file_name:
                        with open(file_path, 'rb') as file:
                            camera_data = pickle.load(file)
                    elif 'label' in file_name:
                        with open(file_path, 'rb') as file:
                            labeled_data = pickle.load(file)
                            # Initialize an empty dictionary to store grouped data
                            grouped_data = defaultdict(list)
                            # Iterate through the labeled_data array
                            for item in labeled_data:
                                key = item[0]  # Extract the first element to be used as the key
                                grouped_data[key].append(item)  # Append the entire tuple to the corresponding key
                            # Convert defaultdict to a regular dictionary
                            labeled_data = dict(grouped_data)

            ####################  Rectify images  ##############  
            # Camera              
            newK = gen_camera_data(camera_data, os.path.join(root,'lei_camera_rectify'), K, dist)
            ####################  Undistort bbox points  ##############
            # Label              
            gen_label_data(labeled_data,  os.path.join(root,'lei_label_rectify'), K, dist, newK)

    
    
    # # Camera
    # keys = list(camera_data.keys())
    # cam_dets = camera_data[keys[0]]['detections'] #['box']
    # image = camera_data[keys[0]]['image_pixels']
    # # Display the image
    # plt.imshow(image)
    # plt.axis('off')  # Optional: Turn off axis labels
    # plt.show()
    
    # # Lidar
    # keys = list(lidar_data.keys())
    # lid_dets = lidar_data[keys[0]]['bboxes'] #['bbox']
    # point_cloud = lidar_data[keys[0]]['points']
    
    
    # # labeled
    # keys = list(labeled_data.keys())
    # idx = 0
    # cam_key = labeled_data[keys[0]][idx][0]
    # cam_box = labeled_data[keys[0]][idx][2]
    # lid_key = labeled_data[keys[0]][idx][1]
    # lid_box = labeled_data[keys[0]][idx][3] #['bbox'] ['object_id']
    # keys = list(labeled_data.keys())



    








