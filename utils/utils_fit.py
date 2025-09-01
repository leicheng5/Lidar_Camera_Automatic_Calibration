import os
import torch
from tqdm import tqdm
from collections import OrderedDict
     
   
def fit_one_epoch(model_train, ema, loss_func, loss_func_cls, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0
    a, b, c     = 5, 1, 1 #Balancing Losses with different weights


    print('Start Train')
    pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    
    model_train.train()
    
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, lidars, img_cents, lid_cents, labels, img_clss, lid_clss = batch
        with torch.no_grad():
            if cuda:
                images    = images.cuda(local_rank)
                # Move each tensor in the lidars list to the GPU
                #lidars = [lidar.cuda(local_rank) for lidar in lidars]
                lidars    = lidars.cuda(local_rank)
                img_cents = img_cents.cuda(local_rank)
                lid_cents = lid_cents.cuda(local_rank)
                labels    = labels.cuda(local_rank)
                img_clss  = img_clss.cuda(local_rank)
                lid_clss  = lid_clss.cuda(local_rank)

        #----------------------#
        #   Zero gradient
        #----------------------#
        optimizer.zero_grad()
        #----------------------#
        #   Forward Propagation
        #----------------------#
        outputs, img_cls_res, lid_cls_res, l3_points = model_train(images, lidars, img_cents, lid_cents)
        loss_value1 = loss_func(outputs.squeeze(1), labels)
        loss_cls1 = loss_func_cls(img_cls_res, img_clss)
        loss_cls2 = loss_func_cls(lid_cls_res, lid_clss)      
        loss_value = a*loss_value1 + b*loss_cls1 + c*loss_cls2
        #----------------------#
        #   Back Propagation
        #----------------------#
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
        # Update parameters
        optimizer.step()
        


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
                                ('lr', optimizer.param_groups[0]['lr']),
                                ('loss_1', loss_value1.item()),
                                ('img_cls', loss_cls1.item()),
                                ('lid_cls', loss_cls2.item())
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
        images, lidars, img_cents, lid_cents, labels, img_clss, lid_clss = batch
        with torch.no_grad():
            if cuda:
                images    = images.cuda(local_rank)
                # Move each tensor in the lidars list to the GPU
                #lidars = [lidar.cuda(local_rank) for lidar in lidars]
                lidars    = lidars.cuda(local_rank)
                img_cents = img_cents.cuda(local_rank)
                lid_cents = lid_cents.cuda(local_rank)
                labels    = labels.cuda(local_rank)
                img_clss  = img_clss.cuda(local_rank)
                lid_clss  = lid_clss.cuda(local_rank)
            #----------------------#
            #   Zero gradient
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   Forward Propagation
            #----------------------#
            outputs, img_cls_res, lid_cls_res, l3_points  = model_train_eval(images, lidars, img_cents, lid_cents)
            loss_value1  = loss_func(outputs.squeeze(1), labels)
            loss_cls1 = loss_func_cls(img_cls_res, img_clss)
            loss_cls2 = loss_func_cls(lid_cls_res, lid_clss)
            #loss_value = a*loss_value1 + b*loss_cls1 + c*loss_cls2
            loss_value = loss_value1 # may consider only common feat loss

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