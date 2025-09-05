import os
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class KeypointDataset(Dataset):
    def __init__(self, root_dir, image_size, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.keypoint_files = []

        # Assuming data is in a structure like root_dir/folder/image.png
        # We need to find all image and corresponding keypoint files.
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith("r.png"):
                        image_path = os.path.join(folder_path, file_name)
                        keypoint_path = image_path.replace("r.png", "cpos.txt")
                        if os.path.exists(keypoint_path):
                            self.image_files.append(image_path)
                            self.keypoint_files.append(keypoint_path)

        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load keypoints, assuming they are in pairs of x, y
        keypoint_path = self.keypoint_files[idx]
        keypoints = []
        with open(keypoint_path, 'r') as f:
            lines = f.readlines()
            # Read only the first 4 keypoints as requested
            for i in range(min(4, len(lines))):
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    keypoints.append((float(parts[0]), float(parts[1])))

        # Apply transformations if provided
        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed['image']
            keypoints = transformed['keypoints']

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        keypoints = torch.tensor(keypoints, dtype=torch.float32) / self.image_size

        return image, keypoints
