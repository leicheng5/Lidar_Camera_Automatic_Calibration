# Lidar_Camera_Automatic_Calibration
Code for our paper: **CalibRefine: Deep Learning-Based Online Automatic Targetless LiDAR–Camera Calibration with Iterative and Attention-Driven Post-Refinement**

**Preprint**: http://arxiv.org/abs/2502.17648

## Abstract
Accurate multi-sensor calibration is essential for deploying robust perception systems in applications such as autonomous driving and intelligent transportation. Existing LiDAR-camera calibration methods often rely on manually placed targets, preliminary parameter estimates, or intensive data preprocessing, limiting their scalability and adaptability in real-world settings. In this work, we propose a fully automatic, targetless, and online calibration framework, CalibRefine, which directly processes raw LiDAR point clouds and camera images. Our approach is divided into four stages: (1) a Common Feature Discriminator that leverages relative spatial positions, visual appearance embeddings, and semantic class cues to identify and generate reliable LiDAR-camera correspondences, (2) a coarse homography-based calibration that uses the matched feature correspondences to estimate an initial transformation between the LiDAR and camera frames, serving as the foundation for further refinement, (3) an iterative refinement to incrementally improve alignment as additional data frames become available, and (4) an attention-based refinement that addresses non-planar distortions by leveraging a Vision Transformer and cross-attention mechanisms. Extensive experiments on two urban traffic datasets demonstrate that CalibRefine achieves high-precision calibration with minimal human input, outperforming state-of-the-art targetless methods and matching or surpassing manually tuned baselines. Our results show that robust object-level feature matching, combined with iterative refinement and self-supervised attention-based refinement, enables reliable sensor alignment in complex real-world conditions without ground-truth matrices or elaborate preprocessing. 
<p align="center">
  <img src="https://github.com/radar-lab/Lidar_Camera_Automatic_Calibration/blob/main/Videos%20and%20Images/framework.png" width="70%">
</p>


## Results
### I. Demo video for Intersection 1
[![Watch the video](https://img.youtube.com/vi/OgJkBQlzVV4/0.jpg)](https://www.youtube.com/watch?v=OgJkBQlzVV4)



### II. Demo video for Intersection 2
[![Watch the video](https://img.youtube.com/vi/Q5gCdi36C8Y/0.jpg)](https://www.youtube.com/watch?v=Q5gCdi36C8Y)

### III. Demo video for using the camera intrinsic matrix to rectify calibration
[![Watch the video](https://img.youtube.com/vi/4HCM0ObuQl4/0.jpg)](https://www.youtube.com/watch?v=4HCM0ObuQl4)

[![Watch the video](https://img.youtube.com/vi/K-TxdrPlns8/0.jpg)](https://www.youtube.com/watch?v=K-TxdrPlns8)



## LiDAR-Camera Online-Automatic-Targetless Calibration Steps
### 1. Common Feature Discriminator
**`Train.py`** to train the Common Feature Discriminator model.

**`Test.py`** to test the accuracy of the Common Feature Discriminator.

### 2. Coarse Homography Calibration
After training the Common Feature Discriminator, you can now use **`lidar_to_camera_calibration.py`** in the folder **`/calibration`** to do the coarse calibration.

### 3. Iterative Refinement
After obtaining the coarse calibration matrix, you can now use **`Online_Refine_Calibration_IT_finetune.py`** in the folder **`/calibration`** to do the iterative refined calibration.

You may also visualize the calibration results by using **`Video_Visualization.py`** in the folder **`/calibration`**.

### 4. Attention-based Refinement
After doing the coarse calibration matrix, you may keep refining the calibration matrix by performing **Attention-based Refinement** in the folder **`/Attention_refinement_model`**.

**`Train_finetune.py`** to train the Attention model.

**`Test_finetune.py`** to test the performence of the Attention model.

**`Adaptive_Correction.py`** to apply Attention-based Refinement

### 5. Intrinsic-Rectified Calibration

If you have the camera intrinsic matrix, you may choose to do **Intrinsic-Rectified Calibration** instead of **Attention-based Refinement**.

You need to use **`rectify_image_bbox.py`** first to employ the intrinsic matrix to rectify image distortion.

Then, you can repeat the above calibration steps (**Coarse Homography Calibration** and **Iterative Refinement**), where the **`.py`** files ending with **`_rectify`** are specifically designed for this purpose.



