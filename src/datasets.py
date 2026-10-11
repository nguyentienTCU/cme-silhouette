import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

def pad_to_target(img, target_height=832, target_width=832):
    # Replace NaN values with 0
    img = np.nan_to_num(img, nan=0.0)
    if img.ndim == 2:
        h, w = img.shape
        pad_h = target_height - h
        pad_w = target_width - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = np.pad(img, ((top, bottom), (left, right)), mode='constant', constant_values=0)
    elif img.ndim == 3:
        # Assuming channel-first format (C, H, W)
        c, h, w = img.shape
        pad_h = target_height - h
        pad_w = target_width - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = np.pad(img, ((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)
    else:
        raise ValueError("Unsupported image dimensions")
    return padded

class ProcessedSegmentationDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=(832,832), transform=None):
        self.file_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.file_df)

    def __getitem__(self, idx):
        rel_path = self.file_df.iloc[idx]["file_path"]
        file_path = os.path.join(self.root_dir, rel_path)
        sample = np.load(file_path, allow_pickle=True).item()

        # Process input image
        img = sample["input"]
        img = np.nan_to_num(img, nan=0.0)
        img = np.expand_dims(img, axis=0).astype(np.float32)  # shape (1, H, W)

        # Process mask (ensure it's binary)
        mask = sample["mask"]
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # shape (1, H, W)

        # Pad to target size
        img = pad_to_target(img, target_height=self.target_size[0], target_width=self.target_size[1])
        mask = pad_to_target(mask, target_height=self.target_size[0], target_width=self.target_size[1])

        if self.transform:
            img, mask = self.transform(img, mask)

        return torch.tensor(img), torch.tensor(mask)

def build_dataloaders(config):
    dataset = ProcessedSegmentationDataset(
        csv_file=config['csv_file'], root_dir=config['root_dir'],
        target_size=config['target_size'])
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, random_state=config['seed'])
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'])
    return train_loader, val_loader
