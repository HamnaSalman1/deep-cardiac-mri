import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import ACDCDataset
from unet import UNet
from tqdm import tqdm

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Load dataset
# -------------------------
dataset = ACDCDataset("data")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# -------------------------
# Model
# -------------------------
model = UNet(n_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# Dice score function
# -------------------------
def dice_score(pred, target, num_classes=4):
    dice = 0.0
    pred = torch.argmax(pred, dim=1)

    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice += (2.0 * intersection + 1e-8) / (union + 1e-8)

    return dice / (num_classes - 1)


# -------------------------
# Training loop
# -------------------------
epochs = 1   # <<< reduced to 1 epoch for quick saving
best_dice = 0.0

for epoch in range(epochs):
    print(f"\nStarting Epoch {epoch+1}/{epochs}")

    # ---- Training ----
    model.train()
    train_loss = 0.0

    for imgs, masks, _ in tqdm(train_loader):
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    val_dice = 0.0

    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            val_dice += dice_score(outputs, masks).item()

    avg_val_dice = val_dice / len(val_loader)

    print(f"\nEpoch {epoch+1} Completed")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation Dice: {avg_val_dice:.4f}")

    # ---- Save model ----
    torch.save(model.state_dict(), "unet_epoch_1.pth")

    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), "unet_best.pth")
        print("Best model saved!")

print("\nTraining finished.")
print(f"Best Validation Dice: {best_dice:.4f}")
