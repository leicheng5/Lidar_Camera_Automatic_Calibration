# Lidar_Camera_Automatic_Calibration
Code for our paper: **CalibRefine: Deep Learning-Based Online Automatic Targetless LiDAR–Camera Calibration with Iterative and Attention-Driven Post-Refinement**

**Preprint**: http://arxiv.org/abs/2502.17648

## Abstract
Accurate multi-sensor calibration is essential for deploying robust perception systems in applications such as autonomous driving and intelligent transportation. Existing LiDAR-camera calibration methods often rely on manually placed targets, preliminary parameter estimates, or intensive data preprocessing, limiting their scalability and adaptability in real-world settings. In this work, we propose a fully automatic, targetless, and online calibration framework, CalibRefine, which directly processes raw LiDAR point clouds and camera images. Our approach is divided into four stages: (1) a Common Feature Discriminator that leverages relative spatial positions, visual appearance embeddings, and semantic class cues to identify and generate reliable LiDAR-camera correspondences, (2) a coarse homography-based calibration that uses the matched feature correspondences to estimate an initial transformation between the LiDAR and camera frames, serving as the foundation for further refinement, (3) an iterative refinement to incrementally improve alignment as additional data frames become available, and (4) an attention-based refinement that addresses non-planar distortions by leveraging a Vision Transformer and cross-attention mechanisms. Extensive experiments on two urban traffic datasets demonstrate that CalibRefine achieves high-precision calibration with minimal human input, outperforming state-of-the-art targetless methods and matching or surpassing manually tuned baselines. Our results show that robust object-level feature matching, combined with iterative refinement and self-supervised attention-based refinement, enables reliable sensor alignment in complex real-world conditions without ground-truth matrices or elaborate preprocessing. 
<center>
  <img src="https://github.com/radar-lab/TransRAD/blob/main/Figures/Fig.2.png" width=70% />
</center>

## Results
### Demo video for Intersection 1
[![Watch the video](https://img.youtube.com/vi/OgJkBQlzVV4/0.jpg)](https://www.youtube.com/watch?v=OgJkBQlzVV4)


### Demo video for Intersection 2
[![Watch the video](https://img.youtube.com/vi/Q5gCdi36C8Y/0.jpg)](https://www.youtube.com/watch?v=Q5gCdi36C8Y)

### Demo video for using the camera intrinsic matrix to rectify calibration
[![Watch the video](https://img.youtube.com/vi/K-TxdrPlns8/0.jpg)](https://www.youtube.com/watch?v=K-TxdrPlns8)


## 
