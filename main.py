import os
import timm  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import random
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import mean_absolute_error

# Fixed Albumentations version warning
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

train_df_path = ""
test_df_path = ""
imgs_path = ""


# Enhanced Dataset with Metadata
class SolarPanelDataset(Dataset):
    def __init__(self, dataframe, transform=None, to_train=True):
        self.dataframe = dataframe
        self.transform = transform
        self.to_train = to_train
        self.placement_map = {
            "roof": 0,
            "openspace": 1,
            "r_openspace": 2,
            "S-unknown": 3,
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = cv2.imread(row["path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Correct color conversion
        image = Image.fromarray(image)

        # Improved metadata encoding
        metadata = torch.zeros(5)
        metadata[0] = 1.0 if row["img_origin"] == "D" else 0.0
        placement = self.placement_map.get(row["placement"], 3)
        metadata[1 + placement] = 1.0  # One-hot encoding

        if self.transform:
            image = self.transform(image)

        if self.to_train:
            target = torch.tensor(
                [row["boil_nbr"], row["pan_nbr"]], dtype=torch.float32
            )
            return image, metadata, target
        return image, metadata


# Model with Metadata
class EfficientNetV2Meta(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b7", pretrained=True, num_classes=0
        )  # you can even try Larger backbone
        self.meta_processor = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
        )
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.regressor = nn.Sequential(
            nn.Linear(self.backbone.num_features + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
            nn.Softplus(),  # Better for count predictions
        )

    def forward(self, image, metadata):
        img_features = self.backbone(image)
        meta_features = self.meta_processor(metadata.unsqueeze(0))
        attn_output, _ = self.attention(meta_features, meta_features, meta_features)
        combined = torch.cat([img_features, attn_output.squeeze(0)], dim=1)
        return self.regressor(combined)


# Custom Transforms
class RandomRotate90:
    def __call__(self, img):
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            return TF.rotate(img, angle)
        return img


class RandomGaussianBlur:
    def __init__(self, p=0.5):
        self.p = p
        self.kernel_sizes = [3, 5, 7]

    def __call__(self, img):
        if random.random() < self.p:
            return TF.gaussian_blur(img, random.choice(self.kernel_sizes))
        return img


# Updated Transforms
train_transform = T.Compose(
    [
        T.RandomResizedCrop(512, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        RandomRotate90(),
        RandomGaussianBlur(p=0.3),
        T.RandomApply([T.ColorJitter(hue=0.2, saturation=0.3, brightness=0.2)], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = T.Compose(
    [
        T.Resize(512),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Training Configuration
def train(fold=0, epochs=20, batch_size=16):
    train_df = pd.read_csv(train_df_path)
    train_df = (
        train_df.groupby("ID")
        .agg(
            {
                "boil_nbr": "sum",
                "pan_nbr": "sum",
                "img_origin": "first",
                "placement": "first",
            }
        )
        .reset_index()
    )
    train_df["path"] = imgs_path + train_df["ID"] + ".jpg"
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(train_df))
    train_idx, val_idx = splits[fold]
    train_ds = SolarPanelDataset(train_df.iloc[train_idx], transform=train_transform)
    val_ds = SolarPanelDataset(train_df.iloc[val_idx], transform=test_transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )
    model = EfficientNetV2Meta().cuda()
    criterion = nn.HuberLoss(delta=1.0)  # Improved loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    scaler = GradScaler()
    best_mae = float("inf")
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, meta, targets in pbar:
            images = images.cuda(non_blocking=True)
            meta = meta.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images, meta)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Validation loop
        model.eval()
        val_loss = 0.0
        preds, truths = [], []
        with torch.no_grad():
            for images, meta, targets in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"
            ):
                images = images.cuda(non_blocking=True)
                meta = meta.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                with autocast():
                    outputs = model(images, meta)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                truths.append(targets.cpu().numpy())

        # Metrics calculation
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        mae = mean_absolute_error(truths, preds)

        print(f"Epoch {epoch+1}/{epochs}")
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {mae:.4f}"
        )

        # Model checkpointing based on MAE
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")

        scheduler.step()

    return best_mae


# Inference 
def predict(test_df, model_paths, batch_size=32):
    test_df["path"] = imgs_path + test_df["ID"] + ".jpg"
    test_ds = SolarPanelDataset(test_df, transform=test_transform, to_train=False)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    predictions = np.zeros((len(test_df), 2))
    for path in model_paths:
        model = EfficientNetV2Meta().cuda()
        model.load_state_dict(torch.load(path, weights_only=True))  # Safer loading
        model.eval()

        tta_preds = []
        with torch.no_grad():
            for images, meta in tqdm(test_loader, desc="Inference"):
                images = images.cuda()
                meta = meta.cuda()
                with autocast():
                    outputs = model(images, meta)
                tta_preds.append(outputs.cpu().numpy())

        predictions += np.concatenate(tta_preds)

    return predictions / len(model_paths)


# Main Execution
if __name__ == "__main__":
    # Train multiple folds
    folds = 5
    model_paths = []
    for fold in range(folds):
        print(f"Training fold {fold+1}/{folds}")
        best_mae = train(fold=fold, epochs=52, batch_size=32)
        model_paths.append(f"best_model_fold{fold}.pth")

    # Prepare submission
    test_df = pd.read_csv(test_df_path)
    predictions = predict(test_df, model_paths, batch_size=64)

    # Create submissions
    submission = pd.DataFrame(
        {"ID": np.repeat(test_df["ID"].values, 2), "Target": predictions.flatten()}
    )
    submission["ID"] += np.where(
        submission.groupby("ID").cumcount() == 0, "_boil", "_pan"
    )
    submission.to_csv("submission_original_b7.csv", index=False)

    int_submission = submission.copy()
    int_submission["Target"] = np.round(int_submission["Target"]).astype(int)
    int_submission.to_csv("submission_integer_b7.csv", index=False)

    print("Submissions saved with shapes:", submission.shape, int_submission.shape)
