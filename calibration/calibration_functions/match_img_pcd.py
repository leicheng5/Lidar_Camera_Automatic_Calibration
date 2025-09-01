# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:32:44 2024

@author: leicheng
"""

import os
import re
import numpy as np

from find_nearest_betw_arr import find_nearest_betw_arr

def extract_and_save_sorted_filenames(input_path, output_path, fn='sorted_filenames.txt'):
    # Extract filenames from the input directory
    filenames = os.listdir(input_path)
    
    # Extract numeric parts from filenames and convert them to integers
    num_filenames = []
    for filename in filenames:
        match = re.match(r'(\d+)\.\d+', filename)
        if match:
            num_filenames.append(match.group(0))
    
    # Sort the files
    num_filenames.sort(key=float)
    
    # Save the sorted array to a file in the output directory
    output_file = os.path.join(output_path, fn)
    with open(output_file, 'w') as f:
        for num in num_filenames:
            f.write(f"{num}\n")
    
    print(f"Sorted filenames saved to {output_file}")

def match_files(input_nobg_dir, input_pcd_dir, input_cam_dir, output_dir, temp_cali_diff=0, fn='matched_files.txt'):
    '''
    Extract filenames from the three directories: input_nobg_dir, input_pcd_dir, and input_cam_dir.
    Load the img_filenames and pcd_filenames.
    Use the find_nearest_betw_arr function to find the closest matching PCD filenames for the image filenames.
    For each match, find the corresponding NOBG file.
    Write the results to a file with the format [img_filename, pcd_filename, nobg_filename].

    '''
    img_filenames = np.loadtxt(os.path.join(output_dir, 'img_filenames.txt'), dtype=str)
    pcd_filenames = np.loadtxt(os.path.join(output_dir, 'pcd_filenames.txt'), dtype=str)
    indices, nearest_pcd_fns = find_nearest_betw_arr(pcd_filenames, img_filenames.astype(float)+temp_cali_diff, thr=0.1)  ## based on img to find pcd
    
    nobg_files = os.listdir(input_nobg_dir)
    pcd_files = os.listdir(input_pcd_dir)
    cam_files = os.listdir(input_cam_dir)
    
    with open(os.path.join(output_dir, fn), 'w') as f:
        for img_fn, pcd_fn, idx in zip(img_filenames, nearest_pcd_fns, indices):
            if np.isnan(pcd_fn): #pcd_fn is None
                continue
            img_filename = f"{img_fn}.png"  #f"{img_fn:.9f}.png"
            pcd_filename = f"{pcd_filenames[idx]}.pcd"
            nobg_filename = f"{pcd_filenames[idx]}_no_bg.pcd"
            
            if img_filename in cam_files and pcd_filename in pcd_files and nobg_filename in nobg_files:
                f.write(f"{img_filename}, {pcd_filename}, {nobg_filename}\n")



if __name__ == "__main__":
    input_nobg_dir = r'E:\FHWA\voxel_detector\outputs'
    output_directory = r'E:\FHWA\calibration\outputs'
    # # Extract and save sorted filenames from 'input_directory' to 'output_directory'
    input_cam_dir = r'E:\FHWA\camera_label\camera'
    #extract_and_save_sorted_filenames(input_cam_dir, output_directory, fn='img_filenames.txt')
    
    input_pcd_dir = r'E:\FHWA\lidar_pcds\lidar_pcds_1'
    #extract_and_save_sorted_filenames(input_pcd_dir, output_directory, fn='pcd_filenames.txt')
    
    temp_cali_diff = 1699029562.279724000-1699029559.080000000 # pcd - img -->3.2
    match_files(input_nobg_dir, input_pcd_dir, input_cam_dir, output_directory, temp_cali_diff, fn='matched_files.txt')
