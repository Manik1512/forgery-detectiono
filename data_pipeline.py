import torch
from torch.utils.data import Dataset,DataLoader,Subset
import pandas as pd
import os 
from PIL import Image
import  albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt 
import cv2
import numpy as np

train_path = r"/home/manik/Documents/datasets/casia_dataset/train"
val_path = r"/home/manik/Documents/datasets/casia_dataset/val"


def get_image_mask_dataframe(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])
    return pd.DataFrame({'image': image_paths, 'mask': mask_paths})

def display_samples(data_laoder):

    batch = next(iter(train_dataloader))
    img, mask = batch  # img: (batch_size, C, H, W), mask: (batch_size, 1, H, W)

    # Extract a single image and mask from the batch
    img = img[0]  # Shape: (C, H, W)
    mask = mask[0]
    print(f"img=>{img.shape} || mask=>{mask.shape}")
  
    # mask = mask.numpy() 
    
    # # Reverse Normalization (undo A.Normalize)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # img = (img * std) + mean  # De-normalize
    # img = np.clip(img, 0, 1)  # Ensure values are in valid range [0,1]

    # # Display  
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)  # Corrected RGB colors
    # plt.title("Image")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(mask, cmap="gray")
    # plt.title("Mask")
    # plt.axis("off")

    # plt.show()

class CustomDataset(Dataset):
    def __init__(self,images,masks,transform):
        self.images=images
        self.masks=masks
        self.transform=transform

    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image_path, mask_path = self.images[idx], self.masks[idx]

        # print(f"idx=>{idx}||{image_path}")
        # print(f"idx=>{idx}||{mask_path}")

        image=cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask=cv2.imread(mask_path)
        mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        mask = np.expand_dims(mask, axis=-1)
        # mask = mask.p   ermute(2, 0, 1)  # Moves the last dimension to the first
        
        if self.transform:
            augmented=self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            mask = (mask > 0.5).float()  #image is normalised , pixels>0.5=>1 else 0
            mask = mask.permute(2, 0, 1)
        return image,mask



transform = A.Compose([
    A.Resize(128, 128),  # Resize both image & mask
    A.HorizontalFlip(p=0.5),  # Flip both
    A.VerticalFlip(p=0.5),  # Flip both
    A.RandomBrightnessContrast(p=0.2),  # Apply brightness/contrast adjustment only to image
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.2),  # Random color jitter
    A.Rotate(limit=45, p=0.5),  # Rotate image within a limit of 45 degrees
    A.GaussianBlur(blur_limit=3, p=0.2),  # Apply Gaussian Blur to image
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image (ImageNet stats)
    ToTensorV2()  # Convert both image and mask to tensors
])


val_transform=A.Compose([
    A.Resize(128,128),  # Resize both image & mask
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image only (sirf 3 channels per hi lagti hai)
    ToTensorV2()  # Convert both image and mask to tensors
])

def data_pipeline(train_path,val_path,transform=transform,val_transform=val_transform,batch=16):

    train_df = get_image_mask_dataframe(os.path.join(train_path, "image"), os.path.join(train_path, "masks"))
    val_df = get_image_mask_dataframe(os.path.join(val_path, "image"), os.path.join(val_path, "masks"))
    
    train_dataset=CustomDataset(train_df["image"],train_df["mask"],transform)
    val_dataset=CustomDataset(val_df["image"],val_df["mask"],val_transform)

    indices=np.random.choice(len(train_dataset),len(train_dataset)//3,replace=False)
    subset=Subset(train_dataset,indices)

    train_dataloader=DataLoader(subset,batch_size=batch,shuffle=True,num_workers=8,pin_memory=True)
    val_dataloader=DataLoader(val_dataset,batch_size=batch,shuffle=True,num_workers=8,pin_memory=True)

    return train_dataloader,val_dataloader

train_dataloader,val_dataloader=data_pipeline(train_path,val_path,transform,val_transform,16)
# display_samples(train_dataloader)

print(len(train_dataloader))

