import os, cv2
import torch
from tqdm import tqdm
from collections import OrderedDict

Zero_Matrix_Test = False

def lid2cam_proj(lid_points, homography_matrix):
    """
    Apply the homography perspective transformation to LiDAR points to project them into the camera frame.
    
    Args:
        lid_points (Tensor or ndarray): Shape [B, N, 2], where B is the batch size and N is the number of points.
        homography_matrix (ndarray): Shape [3, 3], the homography matrix.
    
    Returns:
        projected_lid_points (ndarray): Shape [B, N, 2], transformed points in the camera frame.
    """
    # For batch processing
    batch_size = len(lid_points)
    projected_lid_points = []

    for b in range(batch_size):
        # Apply the homography transformation for each batch item
        batch_projected = cv2.perspectiveTransform(lid_points[b].reshape(-1, 1, 2).cpu().numpy(), homography_matrix[b].cpu().detach().numpy())
        projected_lid_points.append(torch.tensor(batch_projected.squeeze(1), dtype=torch.float32, device=lid_points[b].device))
        
    return projected_lid_points  

def torch_lid2cam_proj(lid_points, homography_matrix):
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
   
def fit_one_epoch(model_train, ema, loss_func, homography_matrix, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0
    a, b, c     = 5, 1, 1 #Balancing Losses with different weights


    print('Start Train')
    pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    
    model_train.train()
    
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, lidars, img_cents, lid_cents, lid_cents_proj = batch
        with torch.no_grad():
            if cuda:
                images    = images.cuda(local_rank)
                lidars    = lidars.cuda(local_rank)
                # Move each tensor in the lidars list to the GPU
                #lidars = [lidar.cuda(local_rank) for lidar in lidars]
                img_cents = [a.cuda(local_rank) for a in img_cents]
                lid_cents = [a.cuda(local_rank) for a in lid_cents]
                lid_cents_proj = [a.cuda(local_rank) for a in lid_cents_proj]                


        #----------------------#
        #   Zero gradient
        #----------------------#
        optimizer.zero_grad()
        #----------------------#
        #   Forward Propagation
        #----------------------#
        delta_matrix, intrinsic_matrix = model_train(images, lidars, img_cents, lid_cents, lid_cents_proj)
        
        #----------------------#
        #   Compute Loss
        #----------------------#
        zeros_matrix = torch.zeros_like(delta_matrix)
        org_matrix = homography_matrix + zeros_matrix
        lid_cents_proj_Org = torch_lid2cam_proj(lid_cents, org_matrix)
        
        ### Delta
        delta_matrix = delta_matrix.view(-1, 3, 3)
        #delta_matrix = delta_matrix.cpu().detach().numpy()
        homography_matrix = torch.tensor(homography_matrix, device=delta_matrix.device, dtype=delta_matrix.dtype)
        new_matrix_with_off = homography_matrix + delta_matrix
        lid_cents_proj_New = torch_lid2cam_proj(lid_cents, new_matrix_with_off)
        #lid_cents_proj_New = torch_lid2cam_proj(lid_cents_proj, delta_matrix)
        ## Matrix Regularization loss: By penalizing large values in delta_H, the model learns to make only the necessary adjustments to initial H.
        mat_reg_loss = delta_matrix.abs().mean(dim=(1, 2))
        
        ### Intrinsic
        #new_H = torch.bmm(intrinsic_matrix, homography_matrix)  # Shape: [B, 3, 3]
        #lid_cents_proj_New = torch_lid2cam_proj(lid_cents_proj, new_H)
        #H_identity = torch.eye(3, device=intrinsic_matrix.device)
        #mat_reg_loss = (intrinsic_matrix - H_identity).abs().mean(dim=(1, 2))

        loss_value = loss_func(img_cents, lid_cents_proj_New, mat_reg_loss, lid_cents_proj_Org)

            
            
            
        # # Check if loss has gradients
        # print("Loss requires_grad:", loss_value.requires_grad)
        #----------------------#
        #   Back Propagation
        #----------------------#
        loss_value.backward()
        
        # # # Check gradients of model parameters
        # for name, param in model_train.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.norm()}")
        #     else:
        #         print(f"No gradient for {name}")
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
        
        # Update parameters
        optimizer.step()
        
        # # Verify parameter updates
        # with torch.no_grad():
        #     for name, param in model_train.named_parameters():
        #         if param.grad is not None:
        #             print(f"{name} updated? {torch.any(param.grad != 0)}")
        #         else:
        #             print(f"{name} has no gradient.")
      


        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            # pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
            #                     'lr'    : optimizer.param_groups[0]['lr'],
            #                     'loss1': loss_value1.item(),
            #                     'img_cls': loss_cls1.item(),
            #                     'lid_cls': loss_cls2.item()})
            pbar.set_postfix(OrderedDict([
                                ('loss', loss / (iteration + 1)),
                                ('lr', optimizer.param_groups[0]['lr'])
                            ]))

            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        
        images, lidars, img_cents, lid_cents, lid_cents_proj = batch
        with torch.no_grad():
            if cuda:
                images    = images.cuda(local_rank)
                lidars    = lidars.cuda(local_rank)
                # Move each tensor in the lidars list to the GPU
                #lidars = [lidar.cuda(local_rank) for lidar in lidars]
                img_cents = [a.cuda(local_rank) for a in img_cents]
                lid_cents = [a.cuda(local_rank) for a in lid_cents]
                lid_cents_proj = [a.cuda(local_rank) for a in lid_cents_proj]  

            #----------------------#
            #   Zero gradient
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   Forward Propagation
            #----------------------#
            delta_matrix, intrinsic_matrix = model_train(images, lidars, img_cents, lid_cents, lid_cents_proj)
            
            #----------------------#
            #   Compute Loss
            #----------------------#
            delta_matrix = delta_matrix.view(-1, 3, 3)
            #delta_matrix = delta_matrix.cpu().detach().numpy()
            homography_matrix = torch.tensor(homography_matrix, device=delta_matrix.device)
            new_matrix_with_off = homography_matrix + delta_matrix
            lid_cents_proj_New = torch_lid2cam_proj(lid_cents, new_matrix_with_off)
            
            zeros_matrix = torch.zeros_like(delta_matrix)
            org_matrix = homography_matrix + zeros_matrix
            lid_cents_proj_Org = torch_lid2cam_proj(lid_cents, org_matrix)
            
            # Matrix Regularization loss: By penalizing large values in delta_H, the model learns to make only the necessary adjustments to initial H.
            mat_reg_loss = delta_matrix.abs().mean(dim=(1, 2))
            loss_value = loss_func(img_cents, lid_cents_proj_New, mat_reg_loss, lid_cents_proj_Org)
            #H_identity = torch.eye(3, device=intrinsic_matrix.device)
            #mat_reg_loss = (intrinsic_matrix - H_identity).abs().mean(dim=(1, 2))       

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        #eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   Save weights
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        # else:
        #     save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))