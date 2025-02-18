import torch
from data_pipeline import val_path
import segmentation_models_pytorch as smp
import os 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import random 
import pandas as pd
import matplotlib.pyplot as plt

def load_model():
    # model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=1)
    # checkpoint_path = "checkpoints/checkpoint_epoch_10.pth"

    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=1)
    checkpoint_path = "checkpoints_new/checkpoint_epoch_12.pth"

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

tta_transforms = [
    A.Compose([A.Resize(128, 128), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(128, 128), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), A.HorizontalFlip(p=1), ToTensorV2()]),
    A.Compose([A.Resize(128, 128), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), A.VerticalFlip(p=1), ToTensorV2()]),
]


def preprocess(image_dir, mask_dir, idx):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))])
    df = pd.DataFrame({'image': image_paths, 'mask': mask_paths})
    img_path = df["image"].iloc[idx]
    mask_path = df["mask"].iloc[idx]
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, mask

def predict_mask(image):
    predictions = []
    for transform in tta_transforms:
        augmented = transform(image=image)
        image_tensor = augmented["image"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(image_tensor)
            pred = torch.sigmoid(pred)
            pred = pred.squeeze().cpu().numpy()
        
        predictions.append(pred)
    
    predictions[1] = np.flip(predictions[1], axis=1)
    predictions[2] = np.flip(predictions[2], axis=0)
    final_prediction = np.mean(predictions, axis=0)
    return (final_prediction > 0.5).astype(np.uint8)

num_predictions = 6
fig, axes = plt.subplots(num_predictions, 3, figsize=(10, 10))

for i in range(num_predictions):
    idx = random.randint(0, 5000)
    original_image, original_mask = preprocess(os.path.join(val_path, "image"), os.path.join(val_path, "masks"), idx)
    binary_mask = predict_mask(original_image)
    
    axes[i, 0].imshow(original_image)
    axes[i, 0].set_title("Input Image")
    axes[i, 0].axis("off")
    
    axes[i, 1].imshow(binary_mask, cmap="gray")
    axes[i, 1].set_title("Predicted Mask")
    axes[i, 1].axis("off")
    
    axes[i, 2].imshow(original_mask, cmap="gray")
    axes[i, 2].set_title("Ground Truth Mask")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
