'''
Author: Lei Cheng 
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ResNet.resnet import Bottleneck, ResNet
from pointnet2_cls_ssg import PointNet2

##################################################### positional encoding ########################################
# def get_position_angle_vec(coord, token_len):
#     """Generate a position angle vector for a given coordinate.

#     Args:
#         coord (int): The coordinate (x or y).
#         token_len (int): The length of the positional encoding vector.

#     Returns:
#         np.ndarray: A vector of angles for positional encoding.
#     """
#     return [coord / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

# def get_2D_position_encoding(x, y, token_len):
#     """Generate positional encoding based on 2D coordinates (x, y).

#     Args:
#         x (int): X-coordinate of the pixel.
#         y (int): Y-coordinate of the pixel.
#         token_len (int): Length of the positional encoding vector.

#     Returns:
#         torch.FloatTensor: Positional encoding vector of length `token_len`.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Ensure token_len is even
#     assert token_len % 2 == 0, "token_len must be even."
#     # If x and y are tensors, move them to CPU and convert to NumPy
#     if isinstance(x, torch.Tensor):
#         x = x.cpu().numpy()
#     if isinstance(y, torch.Tensor):
#         y = y.cpu().numpy()
        
#     # Generate positional encoding for x and y coordinates with token_len/2
#     x_encoding = np.array(get_position_angle_vec(x, token_len // 2))
#     y_encoding = np.array(get_position_angle_vec(y, token_len // 2))

#     # Apply sine to x encoding and cosine to y encoding
#     x_sin = np.sin(x_encoding)
#     y_cos = np.cos(y_encoding)

#     # Interleave x_sin and y_cos
#     position_encoding = np.empty(token_len, dtype=float)
#     position_encoding[0::2] = x_sin
#     position_encoding[1::2] = y_cos

#     return torch.FloatTensor(position_encoding).to(device)

# def get_1D_position_encoding(x, token_len):
#     """Generate positional encoding based on 1D coordinate x.

#     Args:
#         x (int): X-coordinate of the pixel.
#         token_len (int): Length of the positional encoding vector.

#     Returns:
#         torch.FloatTensor: Positional encoding vector of length `token_len`.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # If x is tensor, move it to CPU and convert to NumPy
#     if isinstance(x, torch.Tensor):
#         x = x.cpu().numpy()
#     # Generate positional encoding for x with token_len
#     position_encoding = np.array(get_position_angle_vec(x, token_len))

#     # Interleave x_sin and y_cos
#     position_encoding[0::2] = np.sin(position_encoding[0::2])
#     position_encoding[1::2] = np.cos(position_encoding[1::2])

#     return torch.FloatTensor(position_encoding).to(device)

### Batch ##
def get_position_angle_vec(coords, token_len):
    """Generate a position angle vector for given coordinates in a batch.

    Args:
        coords (np.ndarray): The coordinates (x or y), shape (batch_size,).
        token_len (int): The length of the positional encoding vector.

    Returns:
        np.ndarray: A matrix of angles for positional encoding, shape (batch_size, token_len).
    """
    return np.array([
        [coord / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]
        for coord in coords
    ])

def get_2D_position_encoding(x, y, token_len):
    """Generate positional encoding based on 2D coordinates (x, y) for a batch.

    Args:
        x (np.ndarray or torch.Tensor): X-coordinates of the pixels, shape (batch_size,).
        y (np.ndarray or torch.Tensor): Y-coordinates of the pixels, shape (batch_size,).
        token_len (int): Length of the positional encoding vector.

    Returns:
        torch.FloatTensor: Positional encoding matrix of shape (batch_size, token_len).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure token_len is even
    assert token_len % 2 == 0, "token_len must be even."

    # Convert tensors to NumPy arrays if necessary
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Generate positional encoding for x and y coordinates with token_len/2
    x_encoding = get_position_angle_vec(x, token_len // 2)
    y_encoding = get_position_angle_vec(y, token_len // 2)

    # Apply sine to x encoding and cosine to y encoding
    x_sin = np.sin(x_encoding)
    y_cos = np.cos(y_encoding)

    # Interleave x_sin and y_cos
    position_encoding = np.empty((x.shape[0], token_len), dtype=float)
    position_encoding[:, 0::2] = x_sin
    position_encoding[:, 1::2] = y_cos

    return torch.FloatTensor(position_encoding).to(device)

def get_1D_position_encoding(x, token_len):
    """Generate positional encoding based on 1D coordinate x for a batch.

    Args:
        x (np.ndarray or torch.Tensor): X-coordinates of the pixels, shape (batch_size,).
        token_len (int): Length of the positional encoding vector.

    Returns:
        torch.FloatTensor: Positional encoding matrix of shape (batch_size, token_len).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert tensor to NumPy array if necessary
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    # Generate positional encoding for x with token_len
    position_encoding = get_position_angle_vec(x, token_len)

    # Apply sine and cosine to the position encoding
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    return torch.FloatTensor(position_encoding).to(device)
##################################################### positional encoding END ########################################




class Comm_Feat_Discriminator(nn.Module):
    def __init__(self, num_class=10, token_len = 256):
        super(Comm_Feat_Discriminator, self).__init__()
        self.num_class = num_class  
        self.token_len = token_len
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(160, 160)
        self.bn2 = nn.BatchNorm1d(160)
        self.fc3 = nn.Linear(self.token_len*2, self.token_len*2)
        self.bn3 = nn.BatchNorm1d(self.token_len*2)
        
        self.fc4 = nn.Linear(512+160+self.token_len*2, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.drop4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)        
        self.drop5 = nn.Dropout(0.4)
        self.fc6 = nn.Linear(256, 1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        #self.resnet    = ResNet(Bottleneck, [3, 4, 6, 3], num_class)  #ResNet50()
        self.resnet    = ResNet(Bottleneck, [2, 2, 2, 2], num_class)  #ResNet18()
        self.pointnet2 = PointNet2(num_class, normal_channel=False)

    def forward(self, image_data, lidar_data, image_pos, lidar_pos):  
        img_embed_feat, img_cls_feat, img_cls_res = self.resnet(image_data)
        img_pos_feat        = get_2D_position_encoding(image_pos[:,0], image_pos[:,1], self.token_len)
        
        lidar_data          = lidar_data.permute(0, 2, 1)
        lidar_embed_feat, lidar_cls_feat, lidar_cls_res, l3_points = self.pointnet2(lidar_data)
        lidar_pos_feat      = get_2D_position_encoding(lidar_pos[:,0], lidar_pos[:,1], self.token_len)
        
        # img_embed_feat, img_cls_feat, img_pos_res       = [B, 512], [B, 80], [B, 256]
        # lidar_embed_feat, lidar_cls_feat, lidar_pos_res = [B, 512], [B, 80], [B, 256]
        embed_feat = torch.cat([img_embed_feat, lidar_embed_feat], dim=1)
        #cls_feat   = torch.cat([img_cls_feat, lidar_cls_feat], dim=1)
        cls_feat   = torch.cat([img_cls_res, lidar_cls_res], dim=1)
        pos_feat   = torch.cat([img_pos_feat, lidar_pos_feat], dim=1)

    
        embed_feat = self.drop1(F.relu(self.bn1(self.fc1(embed_feat)))) #[B, 512]
        cls_feat   = self.drop1(F.relu(self.bn2(self.fc2(cls_feat)))) #[B, 160]
        pos_feat   = self.drop1(F.relu(self.bn3(self.fc3(pos_feat)))) #[B, 512]
        
        feats = torch.cat([embed_feat, cls_feat, pos_feat], dim=1)
        feats = self.drop4(F.relu(self.bn4(self.fc4(feats)))) #[B, 512]
        feats = self.drop5(F.relu(self.bn5(self.fc5(feats)))) #[B, 256]
        
        out = self.fc6(feats) #[B, 1]
        
        #out = torch.sigmoid(out)  # for torch.nn.BCELoss()

        return out, img_cls_res, lidar_cls_res, l3_points



def test():
    net = Comm_Feat_Discriminator()
    out, img_cls_res, lidar_cls_res, l3_points = net(torch.randn(1, 3, 32, 32),torch.randn(1, 3, 32, 32))
    print(out.size())

# test()
