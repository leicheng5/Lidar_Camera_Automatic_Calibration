# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:23:09 2024

@author: leicheng
"""
import numpy as np

def find_nearest_betw_arr(known_array, match_array, thr=0.1):
    '''
    Based on match_array, to find the value in an known_array which is closest to an element in match_array
    return match_value and inx in known_array, and arr size is len(match_array)
    '''
    # known_array=np.array([1, 9, 33, 26,  5 , 0, 18, 11])
    # match_array=np.array([-1, 0, 11, 15, 33, 35,10,31])
    # Convert the array to float type
    known_array = known_array.astype(float)
    match_array = match_array.astype(float)
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted] 
    idx = np.searchsorted(known_array_sorted, match_array)
    idx1 = np.clip(idx, 0, len(known_array_sorted)-1)
    idx2 = np.clip(idx - 1, 0, len(known_array_sorted)-1)
    diff1 = known_array_sorted[idx1] - match_array
    diff2 = match_array - known_array_sorted[idx2]
    indices = index_sorted[np.where(diff1 <= diff2, idx1, idx2)]
    
    # Calculate distances and apply the threshold
    nearest_values = known_array[indices]
    distances = np.abs(nearest_values - match_array)
    nearest_values[distances > thr] = None
    
    return indices, nearest_values