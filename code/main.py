from glob import glob
import time
import torch
import utils
from dataloader import GetData
from loss import DiceLoss
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import statistics as s
from torchsummary import summary
import numpy as np

from Unet import UNet, UNet_less

if __name__ == '__main__':
    """ Seeding """
    utils.seeding(28)
    
    """ Hyperparameters """ 
    batch_size = 8
    num_epochs = 100
    lr = 1e-4
    
    """ model """
    device = torch.device('cuda')
    model = UNet_less()
    model = model.to(device)
    
    """ Path """
    checkpoint_path = 'checkpoint/smaller/unet_less/checpoint.pth' # change this 
    results = 'C:/Users/C21048176/Brain Extraction/nifti_results/smaller/unet_less/training/'
    
    """Transform"""
    train_transform = A.Compose(
        [
            A.Rotate(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ElasticTransform(p=0.5),
            A.Normalize(mean=(0,0,0), 
                        std=(1,1,1),max_pixel_value=255.0),
            ToTensorV2()
            ])
    
    transform = A.Compose(
        [
            A.Normalize(mean=(0,0,0), 
                        std=(1,1,1),max_pixel_value=255.0),
            ToTensorV2()
            ])
    
    """ Dataset """
    t1_trainx = sorted(glob('C:/Users/C21048176/Brain Extraction/smaller_dataset/train/images/*'))
    t1_trainy = sorted(glob('C:/Users/C21048176/Brain Extraction/smaller_dataset/train/labels/*'))
    
    t1_valx = sorted(glob('C:/Users/C21048176/Brain Extraction/smaller_dataset/validate/images/*'))
    t1_valy = sorted(glob('C:/Users/C21048176/Brain Extraction/smaller_dataset/validate/labels/*'))
    """ 
    t1_trainx = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T1/train/images/*'))
    t1gd_trainx = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T1Gd/train/images/*'))
    t2_trainx = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T2/train/images/*'))
    flair_trainx = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/Flair/train/images/*'))
    
    t1_trainy = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T1/train/labels/*'))
    t1gd_trainy = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T1Gd/train/labels/*'))
    t2_trainy = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T2/train/labels/*'))
    flair_trainy = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/Flair/train/labels/*'))
    
    t1_valx = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T1/validation/images/*'))
    t1gd_valx = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T1Gd/validation/images/*'))
    t2_valx = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T2/validation/images/*'))
    flair_valx = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/Flair/validation/images/*'))
    
    t1_valy = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T1/validation/labels/*'))
    t1gd_valy = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T1Gd/validation/labels/*'))
    t2_valy = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/T2/validation/labels/*'))
    flair_valy = sorted(glob('C:/Users/C21048176/Brain Extraction/Dataset/Flair/validation/labels/*'))
    """
    
    """ Load dataset """
    train_x = t1_trainx #+ t1gd_trainx + t2_trainx + flair_trainx
    train_y = t1_trainy #+ t1gd_trainy + t2_trainy + flair_trainy
    
    valid_x = t1_valx #+ t1gd_valx + t2_valx + flair_valx
    valid_y = t1_valy #+ t1gd_valy + t2_valy + flair_valy
    
    train_dataset = GetData(train_x, train_y, transform=train_transform)
    valid_dataset = GetData(valid_x, valid_y, transform=transform)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
        )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
        )

    data_str = f'Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n'
    print(data_str)
    
    """ Loss and Learning rate """
    train_loss = []
    val_loss = []
    learning_rate = []
    epoch_time = []
    
    summary(model, (3, 240, 240))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           patience=3, threshold=1e-3,
                                                           min_lr=1e-7, verbose=True)
    loss_fn = DiceLoss()
    
    """ Training """
    best_valid_loss = float('inf')
    early_stopper = utils.EarlyStopper(patience=6)
    
    print('Start training')
    for epoch in range(num_epochs):
        start_time = time.time()
        
        trainLoss = utils.training_loop(model, train_loader, optimizer, loss_fn)
        validLoss = utils.validate_loop(model, valid_loader, loss_fn)
        
        scheduler.step(validLoss) 
        
        """ saving the model """
        if validLoss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {validLoss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = validLoss
            torch.save(model.state_dict(), checkpoint_path)
            
        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        epoch_time.append(end_time - start_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {trainLoss:.3f}\n'
        data_str += f'\t Val. Loss: {validLoss:.3f}\n'
        print(data_str)
        
        """ For graph """
        train_loss.append(trainLoss)
        val_loss.append(validLoss)
        
        current_lr = optimizer.param_groups[0]['lr']
        learning_rate.append(current_lr)
        
        """ Early Stop """
        if early_stopper.early_stop(validLoss):             
            break
        
    print(s.mean(epoch_time))
    np.savetxt(results+'epoch_Time.txt', epoch_time)
    """ Plot """
    utils.plot_loss(train_loss, val_loss)
    utils.plot_lr(learning_rate)
    