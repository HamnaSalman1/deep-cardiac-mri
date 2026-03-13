import torch
import matplotlib.pyplot as plt
from dataset import ACDCDataset
from unet import UNet
from torch.utils.data import random_split

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Dataset
# -------------------------
dataset = ACDCDataset("data")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Take one validation sample
img, mask, label = val_dataset[0]

img = img.unsqueeze(0).to(device)  # add batch dimension

# -------------------------
# Load Model
# -------------------------
model = UNet(n_classes=4).to(device)
model.load_state_dict(torch.load("unet_best.pth", map_location=device))
model.eval()

# -------------------------
# Make Prediction
# -------------------------
with torch.no_grad():
    output = model(img)
    prediction = torch.argmax(output, dim=1).squeeze(0).cpu()

# Remove batch + channel for plotting
img = img.squeeze(0).squeeze(0).cpu()
mask = mask.cpu()

# -------------------------
# Plot Results
# -------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("MRI Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(prediction)
plt.title("Model Prediction")
plt.axis("off")

plt.tight_layout()
plt.show()
