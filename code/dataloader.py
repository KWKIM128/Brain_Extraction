import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class GetData(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_path[idx]).convert('RGB'), 
                         dtype=np.float32)
        mask = np.array(Image.open(self.mask_path[idx]).convert('L'), 
                         dtype=np.float32)
        mask = (mask/255).astype(np.float32)
        
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        
        return image, mask
