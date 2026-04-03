# SCVD_CNN_LSTM_Training_Windows.py

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
import numpy as np

# ----------------------
# 1. Dataset Class
# ----------------------
class SCVDDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, transform=None):
        self.root = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.samples = []

        classes = sorted(os.listdir(root_dir))
        self.class2idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            cdir = os.path.join(root_dir, c)
            for fname in os.listdir(cdir):
                if fname.endswith((".mp4", ".avi")):
                    path = os.path.join(cdir, fname)
                    self.samples.append((path, self.class2idx[c]))

    def __len__(self):
        return len(self.samples)

    def read_clip(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < self.clip_len:
            indices = np.linspace(0, total-1, total).astype(int)
        else:
            indices = np.linspace(0, total-1, self.clip_len).astype(int)
        idx_set = set(indices)
        cnt = 0
        ret = True
        while ret and len(frames) < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            if cnt in idx_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cnt += 1
        cap.release()
        # pad if frames < clip_len
        while len(frames) < self.clip_len:
            frames.append(frames[-1])
        return frames

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self.read_clip(path)
        clip = []
        for frame in frames:
            img = T.ToPILImage()(frame)
            if self.transform:
                img = self.transform(img)
            clip.append(img)
        clip = torch.stack(clip, dim=0)  # (T, C, H, W)
        return clip, label


# ----------------------
# 2. CNN + LSTM Model
# ----------------------
class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=256):
        super().__init__()
        # Pretrained ResNet18 as feature extractor
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # remove fc layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = self.feature_extractor(x[:, t])  # (b, 512, 1, 1)
            f = f.view(b, -1)                    # (b, 512)
            feats.append(f)
        feats = torch.stack(feats, dim=1)        # (b, T, 512)
        out, _ = self.lstm(feats)               # (b, T, hidden_dim)
        last = out[:, -1, :]                     # last time step
        logits = self.classifier(last)          # (b, num_classes)
        return logits


# ----------------------
# 3. Main Training Script
# ----------------------
if __name__ == "__main__":
    # ----------------------
    # 3.1 Transforms
    # ----------------------
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # ----------------------
    # 3.2 Dataset paths (Windows raw strings)
    # ----------------------
    train_dataset = SCVDDataset(r"D:\archive (5)\SCVD\SCVD_converted\Train", clip_len=16, transform=transform)
    val_dataset = SCVDDataset(r"D:\archive (5)\SCVD\SCVD_converted\Test", clip_len=16, transform=transform)

    print("Number of training videos:", len(train_dataset))
    print("Number of validation videos:", len(val_dataset))

    # ----------------------
    # 3.3 DataLoaders
    # ----------------------
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # ----------------------
    # 3.4 Device, model, loss, optimizer
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTMModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ----------------------
    # 3.5 Training Loop
    # ----------------------
    num_epochs = 10  # adjust as needed
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

        # ----------------------
        # Validation
        # ----------------------
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips = clips.to(device)
                labels = labels.to(device)
                outputs = model(clips)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total
        print(f"Validation Acc: {val_acc:.2f}%")
        model.train()

    # ----------------------
    # 3.6 Save Model
    # ----------------------
    torch.save(model.state_dict(), "SCVD_CNN_LSTM.pth")
    print("Training complete! Model saved as SCVD_CNN_LSTM.pth")
