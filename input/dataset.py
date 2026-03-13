import os
import glob
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def read_info(cfg_path):
    info = {}
    with open(cfg_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(": ")
            info[key] = value
    return info


class ACDCDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        # Find all patient folders in training set
        patients = sorted(glob.glob(os.path.join(root_dir, "training", "patient*")))

        for patient in patients:
            frames = glob.glob(os.path.join(patient, "*_frame*.nii.gz"))

            for f in frames:
                if "_gt" not in f:
                    gt = f.replace(".nii.gz", "_gt.nii.gz")

                    if os.path.exists(gt):
                        img_volume = nib.load(f).get_fdata()
                        mask_volume = nib.load(gt).get_fdata()

                        info_path = os.path.join(patient, "Info.cfg")
                        label = read_info(info_path)["Group"]

                        # Convert 3D volume into 2D slices
                        for slice_idx in range(img_volume.shape[2]):
                            img_slice = img_volume[:, :, slice_idx]
                            mask_slice = mask_volume[:, :, slice_idx]

                            self.samples.append((img_slice, mask_slice, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask, label = self.samples[idx]

        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        # Normalize per slice
        img = (img - img.mean()) / (img.std() + 1e-8)

        # Add channel dimension
        img = img.unsqueeze(0)  # (1, H, W)
        mask = mask.unsqueeze(0).float()  # temporarily float for resizing

        # Resize to fixed 256x256
        img = F.interpolate(
            img.unsqueeze(0),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(256, 256),
            mode='nearest'
        ).squeeze(0)

        mask = mask.long().squeeze(0)

        return img, mask, label
