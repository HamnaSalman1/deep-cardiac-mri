from dataset import ACDCDataset

dataset = ACDCDataset("data")

print("Total annotated volumes:", len(dataset))

img, mask, label = dataset[0]

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Pathology label:", label)
