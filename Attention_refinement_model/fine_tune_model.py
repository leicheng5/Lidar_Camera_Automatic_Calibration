'''
Author: Lei Cheng 
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from vit_pytorch.vit import ViT
from pointnet2_cls_ssg_Lei import PointNet2

##################################################### positional encoding ########################################
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

def load_pretrain(model, pretain_path, load_head=False, strict=False):
    """
    load the jax pretrained weights from timm, note that we remove many unnecessary components (e.g., mlp_head) 
    
    weights can be downloaded from here: https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-vitjx
    you can download various pretriained weights and adjust your codes to fit them

    ideas from https://github.com/Sebastian-X/vit-pytorch-with-pretrained-weights/blob/master/tools/trans_weight.py

    weights mapping as follows:
    
    timm_jax_vit_base                           self

    pos_embed                                   pos_embedding
    patch_embed.proj.weight                     to_patch_embedding.0.weights
    patch_embed.proj.bias                       to_patch_embedding.0.bias
    cls_token                                   cls_token
    norm.weight                                 transformer.norm.weight
    norm.bias                                   transformer.norm.bias

                        -----------Attention Layer-------------
    blocks.0.norm1.weight                       transformer.layers.0.0.norm.weight
    blocks.0.norm1.bias                         transformer.layers.0.0.norm.bias
    blocks.0.attn.qkv.weight                    transformer.layers.0.0.to_qkv.weight
    blocks.0.attn.qkv.bias                      transformer.layers.0.0.to_qkv.bias
    blocks.0.attn.proj.weight                   transformer.layers.0.0.to_out.0.weight
    blocks.0.attn.proj.bias                     transformer.layers.0.0.to_out.0.bias
                        -----------MLP Layer-------------
    blocks.0.norm2.weight                       transformer.layers.0.1.net.0.weight
    blocks.0.norm2.bias                         transformer.layers.0.1.net.0.bias
    blocks.0.mlp.fc1.weight                     transformer.layers.0.1.net.1.weight
    blocks.0.mlp.fc1.bias                       transformer.layers.0.1.net.1.bias
    blocks.0.mlp.fc2.weight                     transformer.layers.0.1.net.4.weight
    blocks.0.mlp.fc2.bias                       transformer.layers.0.1.net.4.bias
            .                                                      .
            .                                                      .
            .                                                      .
    """
    jax_dict = torch.load(pretain_path, map_location='cpu')
    new_dict = {}
    #print(jax_dict.keys())
    def add_item(key, value):
        key = key.replace('blocks', 'transformer.layers')
        new_dict[key] = value
        
    for key, value in jax_dict.items():
        if key == 'cls_token':
            new_dict[key] = value
        
        elif 'norm1' in key:
            new_key = key.replace('norm1', '0.norm')
            add_item(new_key, value)
        elif 'attn.qkv' in key:
            new_key = key.replace('attn.qkv', '0.to_qkv')
            add_item(new_key, value)
        elif 'attn.proj' in key:
            new_key = key.replace('attn.proj', '0.to_out.0')
            add_item(new_key, value)
        elif 'norm2' in key:
            new_key = key.replace('norm2', '1.net.0')
            add_item(new_key, value)
        elif 'mlp.fc1' in key:
            new_key = key.replace('mlp.fc1', '1.net.1')
            add_item(new_key, value)
        elif 'mlp.fc2' in key:
            new_key = key.replace('mlp.fc2', '1.net.4')
            add_item(new_key, value)
        elif 'patch_embed.proj' in key:
            new_key = key.replace('patch_embed.proj', 'to_patch_embedding.0')
            add_item(new_key, value)

        
        elif key == 'pos_embed':
            add_item('pos_embedding', value)
        elif key == 'norm.weight':
            add_item('transformer.norm.weight', value)
        elif key == 'norm.bias':
            add_item('transformer.norm.bias', value)
            
        if load_head:    
            # Map head to mlp_head
            if key == 'head.weight':
                new_dict['mlp_head.weight'] = value
            elif key == 'head.bias':
                new_dict['mlp_head.bias'] = value            
        
    model.load_state_dict(new_dict, strict=strict)     


### Cross Attention between lidar and camera
class CrossAttentionRelation(nn.Module):    
    def __init__(self, input_dim, embed_dim, num_heads, output_dim, num_layers=6, feedforward_dim=256, dropout_rate=0.1):
        """
        Initialize the CrossAttentionRelation model, which uses cross-attention 
        to learn the relationship between two inputs (img_cents and lid_cents).
        
        Args:
            input_dim (int): The dimensionality of the input features (e.g., per point feature size).
            embed_dim (int): The embedding dimension used for the attention mechanism.
            num_heads (int): Number of heads for multi-head attention.
            output_dim (int): Dimensionality of the output features (e.g., relation embedding size).
            num_layers (int): Number of stacked attention layers.
            feedforward_dim (int): Dimensionality of the feedforward network.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        self.num_layers = num_layers
        
        self.query_embed = nn.Linear(input_dim, embed_dim)
        self.key_embed = nn.Linear(input_dim, embed_dim)
        self.value_embed = nn.Linear(input_dim, embed_dim)

        #self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Stacked multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Feedforward network with dropout
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Layer normalization
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers + 1)])  # +1 for feedforward
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final fully connected layer
        self.fc = nn.Linear(embed_dim, output_dim)
        
    def pad_sequences(self, data_list):
        """
        Preprocess a list of variable-length tensors into padded tensors and masks.

        Args:
            data_list (list of torch.Tensor): Each tensor has shape [N, D].
        
        Returns:
            padded_data (torch.Tensor): Padded tensor with shape [B, max_N, D].
            mask (torch.BoolTensor): Mask with shape [B, max_N], where True indicates padding.
        """
        # Determine the device from the first tensor in the list
        device = data_list[0].device
        
        # Pad the sequences
        padded_data = pad_sequence(data_list, batch_first=True).to(device)  # [B, max_N, D]
        
        # Create the mask
        lengths = torch.tensor([data.shape[0] for data in data_list], dtype=torch.long, device=device)
        max_length = lengths.max().item()
        mask = (torch.arange(max_length, device=device)[None, :] >= lengths[:, None])  # [B, max_N]
        
        return padded_data, mask    

    def forward(self, img_cents, lid_cents, lid_cents_proj):
        # Padding the input seq data
        img_cents_padded, img_mask = self.pad_sequences(img_cents)  # [B, max_N, 2], [B, max_N]
        lid_cents_padded, lid_mask = self.pad_sequences(lid_cents)  # [B, max_M, 2], [B, max_M]
        lid_cents_proj_padded, lid_proj_mask = self.pad_sequences(lid_cents_proj)  # [B, max_M, 2], [B, max_M]

        # Embed the inputs
        query = self.query_embed(img_cents_padded)  # [B, N, embed_dim]
        key = self.key_embed(lid_cents_proj_padded)      # [B, M, embed_dim]
        value = self.value_embed(lid_cents_padded)  # [B, M, embed_dim]

        # # Cross-attention
        # attn_output, _ = self.attention(
        #     query, key, value, key_padding_mask=lid_mask
        # )  
        
        # Stacked cross-attention layers with residual connections, normalization, and dropout
        for i, attention_layer in enumerate(self.attention_layers):
            attn_output, _ = attention_layer(query, key, value, key_padding_mask=lid_mask)# [B, N, embed_dim]
            query = query + attn_output  # Add residual connection  # [B, N, embed_dim]
            query = self.norm_layers[i](query)  # Apply layer normalization
            #query = self.dropout(query)  # Apply dropout
        
        # Apply feedforward network with residual connection
        feedforward_output = self.feedforward(query)  # [B, N, embed_dim]
        query = query + feedforward_output  # Add residual connection
        query = self.norm_layers[-1](query)  # Apply layer normalization

        # Global pooling (mean over sequence length N, ignoring padded values)
        pooled_output = (attn_output * (~img_mask).unsqueeze(-1)).sum(dim=1)  # Sum non-padded features
        valid_lengths = (~img_mask).sum(dim=1, keepdim=True)  # Number of valid elements
        pooled_output = pooled_output / valid_lengths.clamp(min=1)  # Mean pooling # [B, embed_dim]

        # Fully connected layer for output
        relation = self.fc(pooled_output)  # [B, output_dim]

        return relation

class Out_limit_Acti(nn.Module):
    def __init__(self, range_len):
        super().__init__()
        self.N = range_len  # Desired range limit

    def forward(self, linear_output):
        # Apply tanh to constrain the output to [-1, 1], then scale to [-N, N]
        constrained_output = self.N * torch.tanh(linear_output)
        return constrained_output

### Main Function
class FineTune_Model(nn.Module):
    def __init__(self, pretrain_path, input_shape = [384, 384], num_class=1000, out_dim = 9, token_len = 256, range_len=10):
        super(FineTune_Model, self).__init__()
        self.num_class = num_class  
        self.token_len = token_len
        self.fc0 = nn.Linear(num_class, num_class)
        self.bn0 = nn.BatchNorm1d(num_class)
        self.drop0 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(2*num_class, num_class)
        self.bn1 = nn.BatchNorm1d(num_class)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(num_class, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)        
        self.drop3 = nn.Dropout(0.4)
        
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.drop4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)        
        self.drop5 = nn.Dropout(0.4)
        self.fc6 = nn.Linear(512, out_dim)  #head 1
        self.fc7 = nn.Linear(512, out_dim)  #head 2
        
        self.fc8 = nn.Linear(512, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.drop8 = nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.limit_out = Out_limit_Acti(range_len=range_len)
        
        #self.resnet    = ResNet(Bottleneck, [2, 2, 2, 2], num_class)  #ResNet18()
        
        # vision transformer
        self.vit = ViT(
            num_classes = num_class,
            image_size = (input_shape[0], input_shape[1]),  # image size is a tuple of (height, width)
            patch_size = (16, 16),    # patch size is a tuple of (height, width)
            dim = 768,
            depth = 12,
            heads = 12,
            mlp_dim = 3072,
            dropout = 0.1,
            emb_dropout = 0.1,
            qkv_bias = True,
            conv_patch = True
        )
        
        # Print model's state dictionary keys to see all parameter names
        #print(self.vit.state_dict().keys())
        # Load the pretrained weights
        if pretrain_path is not None:
            load_pretrain(self.vit, pretrain_path)

        # pointnet++
        self.pointnet2 = PointNet2(num_class, normal_channel=False)
        
        # relation between lidar cent_points and camera cent_points
        self.cross_att = CrossAttentionRelation(input_dim=2, embed_dim=512, num_heads=16, output_dim=512, num_layers=6, feedforward_dim=512, dropout_rate=0.1)


    def forward_all(self, image_data, lidar_data, img_cents, lid_cents, lid_cents_proj):
        ## img global feat
        img_global_feat = self.vit(image_data) # [B, num_class]
        ## lidar global feat
        lidar_data          = lidar_data.permute(0, 2, 1)
        lidar_embed_feat, lidar_global_feat, _ = self.pointnet2(lidar_data)# [B, num_class]

        # Concate feat
        # img_global_feat       = [B, num_class]
        # lidar_global_feat     = [B, num_class]
        # global_feat           = [B, 2*num_class]
        global_feat = torch.cat([img_global_feat, lidar_global_feat], dim=1)

        # Distortion Learning
        global_feat = self.drop1(F.relu(self.bn1(self.fc1(global_feat)))) #[B, num_class]
        global_feat = self.drop2(F.relu(self.bn2(self.fc2(global_feat)))) #[B, 512]
        global_feat = self.drop3(F.relu(self.bn3(self.fc3(global_feat)))) #[B, 256]
        
        # Relation Learning
        relation_feat = self.cross_att(img_cents, lid_cents, lid_cents_proj) #[B, 512]
        relation_feat = self.drop8(F.relu(self.bn8(self.fc8(relation_feat)))) #[B, 256]
        
        final_feat = torch.cat([global_feat, relation_feat], dim=1) #[B, 512]
        
        # Two different heads
        delta_matrix = self.fc6(final_feat) #[B, 9]
        #delta_matrix = self.limit_out(delta_matrix)
        intrinsic_matrix = None #self.fc7(final_feat) #[B, 9]

        return delta_matrix, intrinsic_matrix
    
    def forward_img(self, image_data, lidar_data, img_cents, lid_cents, lid_cents_proj):
        ## img global feat
        img_global_feat = self.vit(image_data) # [B, num_class]

        # Concate feat
        # img_global_feat       = [B, num_class]
        # global_feat           = [B, num_class]
        global_feat = img_global_feat

        # Distortion Learning
        global_feat = self.drop0(F.relu(self.bn0(self.fc0(global_feat)))) #[B, num_class]
        global_feat = self.drop2(F.relu(self.bn2(self.fc2(global_feat)))) #[B, 512]
        global_feat = self.drop3(F.relu(self.bn3(self.fc3(global_feat)))) #[B, 256]
        
        # Relation Learning
        relation_feat = self.cross_att(img_cents, lid_cents, lid_cents_proj) #[B, 512]
        relation_feat = self.drop8(F.relu(self.bn8(self.fc8(relation_feat)))) #[B, 256]
        
        final_feat = torch.cat([global_feat, relation_feat], dim=1) #[B, 512]
        
        # Two different heads
        delta_matrix = self.fc6(final_feat) #[B, 9]
        #delta_matrix = self.limit_out(delta_matrix)
        intrinsic_matrix = None #self.fc7(final_feat) #[B, 9]

        return delta_matrix, intrinsic_matrix    
    
    def forward_cents(self, img_cents, lid_cents, lid_cents_proj):
        # Relation Learning
        relation_feat = self.cross_att(img_cents, lid_cents, lid_cents_proj) #[B, 512]
        
        # Distortion Learning
        relation_feat = self.drop4(F.relu(self.bn4(self.fc4(relation_feat)))) #[B, 512]
        relation_feat = self.drop5(F.relu(self.bn5(self.fc5(relation_feat)))) #[B, 512]
        
        
        # Two different heads
        delta_matrix = self.fc6(relation_feat) #[B, 9]
        #delta_matrix = self.limit_out(delta_matrix)
        intrinsic_matrix = None #self.fc7(final_feat) #[B, 9]
        
        return delta_matrix, intrinsic_matrix    

    def forward(self, image_data, lidar_data, img_cents, lid_cents, lid_cents_proj):
        # Relation Learning
        #delta_matrix, intrinsic_matrix = self.forward_cents(img_cents, lid_cents, lid_cents_proj)
        
        # Raw data Feat and Relation Learning
        #delta_matrix, intrinsic_matrix = self.forward_all(image_data, lidar_data, img_cents, lid_cents, lid_cents_proj)
        
        # img Feat and Relation Learning
        delta_matrix, intrinsic_matrix = self.forward_img(image_data, lidar_data, img_cents, lid_cents, lid_cents_proj)
      
        return delta_matrix, intrinsic_matrix 


def test():
    net = FineTune_Model()
    out = net(torch.randn(1, 3, 384, 384),torch.randn(1, 3, 384, 384))
    print(out.size())

# test()
