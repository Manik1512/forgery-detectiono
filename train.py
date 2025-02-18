import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, jaccard_score
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from data_pipeline import *
from metrics import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import csv

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience



def log_metrics(epoch, metrics,train_loss,val_loss, log_file="metrics_log.csv"):
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # If the file doesn't exist, write the header first
        if not file_exists:
            writer.writerow(["epoch", "accuracy", "auc", "f1", "mcc", "iou","train_loss", "val_loss"])
        
        writer.writerow([epoch] + list(metrics.values())+[train_loss, val_loss])


# def compute_metrics(y_true, y_pred):
#     y_true = y_true.flatten()
#     y_pred = (y_pred.flatten() > 0.5).astype(int)
#     return {
#         'Accuracy': accuracy_score(y_true, y_pred),
#         'AUC': roc_auc_score(y_true, y_pred),
#         'F1 Score': f1_score(y_true, y_pred),
#         'MCC': matthews_corrcoef(y_true, y_pred),
#         'IoU': jaccard_score(y_true, y_pred)
    #     }

def compute_metrics(y_true, y_pred):  
# Initialize cumulative variables to store intermediate results
    acc = 0
    auc = 0
    f1 = 0
    mcc = 0
    iou = 0
    total_samples = 0
    print("in compute metrics")
    for true, pred in zip(y_true, y_pred):
        true = true.flatten()  # Flatten per batch
        pred = (pred.flatten() > 0.5).astype(int)  # Flatten and threshold per batch

        # Incrementally update metrics
        acc += accuracy_score(true, pred)
        auc += roc_auc_score(true, pred)
        f1 += f1_score(true, pred)
        mcc += matthews_corrcoef(true, pred)
        iou += jaccard_score(true, pred)
        total_samples += 1
    print("for loop done")
    # Average over batches
    return {
        'Accuracy': acc / total_samples,
        'AUC': auc / total_samples,
        'F1 Score': f1 / total_samples,
        'MCC': mcc / total_samples,
        'IoU': iou / total_samples
    }


def save_checkpoint(model, optimizer, epoch, checkpoint_dir="checkpoints_new"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def dice_loss(pred, target, epsilon=1e-6):
    intersection = 2 * (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (intersection / torch.max(union, torch.tensor(epsilon)))

def focal_loss(pred, target, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)  # Probabilities
    loss = (1 - pt) ** gamma * bce
    return loss.mean()

def total_loss(pred, target, lambda_factor=1.0, gamma=2.0):
    return dice_loss(pred, target) + lambda_factor * focal_loss(pred, target, gamma)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train_model(model, train_loader, val_loader, optimizer, num_epochs=50, patience=5):
    early_stopping = EarlyStopping(patience=patience)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for imgs, masks in train_pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            outputs = torch.sigmoid(outputs)
            loss = total_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
        
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        metrics={}
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for imgs, masks in val_pbar:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                outputs = torch.sigmoid(outputs)
                loss = total_loss(outputs, masks)
                val_loss += loss.item()
                y_true.append(masks.cpu().numpy())
                y_pred.append(outputs.cpu().numpy())
                val_pbar.set_postfix(loss=loss.item())
                
        print("done evaluating val data")
        y_true = np.vstack(y_true)  # Use vstack instead of concatenate
        y_pred = np.vstack(y_pred)  # Use vstack to avoid excessive memory allocation
        print("done stacking")
        metrics = compute_metrics(y_true, y_pred)
        print("done calcualting metrics")
        avg_train_loss=train_loss/len(train_loader)
        avg_val_loss=val_loss/len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(metrics)
        log_metrics(epoch, metrics,avg_train_loss,avg_val_loss)

        save_checkpoint(model, optimizer, epoch)
        scheduler.step(avg_val_loss)
        
        if early_stopping(avg_val_loss):
            print("Early stopping triggered.")
            break
from torch.optim.lr_scheduler import ReduceLROnPlateau





if __name__ == "__main__":
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=1)  #kal mobinenet pe kia tha
    # model = smp.Unet(encoder_name="inceptionresnetv2", encoder_weights="imagenet", in_channels=3, classes=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print("Pretrained U-Net model loaded for binary segmentation.")

    history =train_model(model=model,train_loader=train_dataloader,val_loader=val_dataloader,optimizer=optimizer,num_epochs=12,patience=3)

    