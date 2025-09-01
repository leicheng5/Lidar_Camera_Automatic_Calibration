import torch
import cv2
import numpy as np


def torch_perspective_transform(lid_points, homography_matrix):
    """
    Apply the homography perspective transformation using PyTorch.

    Args:
        lid_points (list of Tensor): List of 2D tensors, each of shape [N, 2], where N is the number of points.
        homography_matrix (Tensor): Tensor of shape [B, 3, 3], homography matrices for each batch.

    Returns:
        list of Tensor: Transformed points in the camera frame, with the same length as lid_points.
    """
    batch_size = len(lid_points)
    projected_lid_points = []

    for b in range(batch_size):
        num_points = lid_points[b].shape[0]  # Number of points for the current batch
        ones = torch.ones((num_points, 1), device=lid_points[b].device, dtype=lid_points[b].dtype)
        
        # Convert points to homogeneous coordinates
        lid_points_h = torch.cat((lid_points[b], ones), dim=-1)  # Shape [N, 3]

        # Perform the transformation
        projected_h = torch.matmul(lid_points_h, homography_matrix[b].T)  # Shape [N, 3]

        # Normalize to get 2D points
        projected_points = projected_h[:, :2] / projected_h[:, 2:3]  # Shape [N, 2]

        # Append results
        projected_lid_points.append(projected_points)

    return projected_lid_points


# def torch_perspective_transform(lid_points, homography_matrix):
#     """
#     Apply the homography perspective transformation using PyTorch.
#     """
#     batch_size, num_points, _ = lid_points.shape
#     ones = torch.ones((batch_size, num_points, 1), device=lid_points.device, dtype=lid_points.dtype)
#     lid_points_h = torch.cat((lid_points, ones), dim=-1)  # Shape [B, N, 3]
#     projected_h = torch.bmm(lid_points_h, homography_matrix.transpose(1, 2))  # Shape [B, N, 3]
#     projected_lid_points = projected_h[:, :, :2] / projected_h[:, :, 2:3]  # Normalize to get 2D points
#     return projected_lid_points

def cv2_perspective_transform(lid_points, homography_matrix):
    """
    Apply the homography perspective transformation using OpenCV.
    """
    batch_size = len(lid_points)
    projected_lid_points = []
    for b in range(batch_size):
        lid_points_np = lid_points[b].cpu().numpy()
        homography_matrix_np = homography_matrix[b].cpu().numpy()
        projected = cv2.perspectiveTransform(lid_points_np.reshape(-1, 1, 2), homography_matrix_np)
        projected_lid_points.append(torch.tensor(projected.squeeze(1), dtype=torch.float32, device=lid_points.device))
    return torch.stack(projected_lid_points)

# Test parameters
batch_size = 2
num_points = 5

# Random points
lid_points = torch.randn(batch_size, num_points, 2, dtype=torch.float32, device='cuda')

# Random homography matrices
homography_matrix = torch.randn(batch_size, 3, 3, dtype=torch.float32, device='cuda')
#homography_matrix[:, 2, 2] = 1.0  # Ensure the bottom-right corner is 1 for a valid homography

# Perform transformations
torch_result = torch_perspective_transform(lid_points, homography_matrix)
cv2_result = cv2_perspective_transform(lid_points, homography_matrix)

# Compare results
print("Torch Result:")
print(torch_result)

print("\nOpenCV Result:")
print(cv2_result)

# Check consistency
difference = torch.abs(torch_result - cv2_result)
print("\nDifference:")
print(difference)

# Verify if the difference is small
tolerance = 1e-4
if torch.all(difference < tolerance):
    print("\nThe results are consistent within the tolerance of", tolerance)
else:
    print("\nThe results are NOT consistent.")
