import csv
import matplotlib.pyplot as plt

# Function to read metrics from the CSV log file
def read_metrics(log_file="metrics_log.csv"):
    epochs = []
    train_loss = []
    val_loss = []
    accuracy = []
    auc = []
    f1 = []
    mcc = []
    iou = []
    
    # Read the CSV file and extract metrics
    with open(log_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            epochs.append(int(row[0]))  # Epoch
            accuracy.append(float(row[1]))  # Accuracy
            auc.append(float(row[2]))  # AUC
            f1.append(float(row[3]))  # F1 score
            mcc.append(float(row[4]))  # MCC
            iou.append(float(row[5]))  # IoU
            train_loss.append(float(row[6]))  # Train loss
            val_loss.append(float(row[7]))  # Validation loss
    
    return epochs, train_loss, val_loss, accuracy, auc, f1, mcc, iou

# Read the metrics from the CSV file
log_file = "metrics_log.csv"
epochs, train_loss, val_loss, accuracy, auc, f1, mcc, iou = read_metrics(log_file)

# Plotting train loss and validation loss on the same graph
plt.subplot(2,3,1)
# plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Train Loss", color='blue', marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", color='red', marker='x')
plt.title("Train Loss vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
# plt.show()

# Plotting other metrics
# Accuracy
plt.subplot(2,3,2)
# plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, label="Accuracy", color='green', marker='o')
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
# plt.show()

# AUC
plt.subplot(2,3,3)
# plt.figure(figsize=(10, 6))
plt.plot(epochs, auc, label="AUC", color='orange', marker='x')
plt.title("AUC (Area Under Curve)")
plt.xlabel("Epochs")
plt.ylabel("AUC")
plt.grid(True)
# plt.show()

# F1 Score
plt.subplot(2,3,4)
# plt.figure(figsize=(10, 6))
plt.plot(epochs, f1, label="F1 Score", color='purple', marker='s')
plt.title("F1 Score")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.grid(True)
# plt.show()

# MCC (Matthews Correlation Coefficient)
plt.subplot(2,3,5)
# plt.figure(figsize=(10, 6))
plt.plot(epochs, mcc, label="MCC", color='brown', marker='^')
plt.title("MCC (Matthews Correlation Coefficient)")
plt.xlabel("Epochs")
plt.ylabel("MCC")
plt.grid(True)
# plt.show()

# IoU (Intersection over Union)
plt.subplot(2,3,6)
# plt.figure(figsize=(10, 6))
plt.plot(epochs, iou, label="IoU", color='cyan', marker='D')
plt.title("IoU (Intersection over Union)")
plt.xlabel("Epochs")
plt.ylabel("IoU")
plt.grid(True)
plt.show()
