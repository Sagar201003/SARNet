import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import classification_report

# ==========================================
# 1. CONFIGURATION (Matches your CycleGAN notebook)
# ==========================================
class Config:
    DATA_ROOT      = '/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2'
    CLASSES        = ['agri', 'barrenland', 'grassland', 'urban']
    
    # Matching your subset constraints
    MAX_TRAIN_SAMPLES = 2000  # Total subset for training
    MAX_VAL_SAMPLES   = 200   # Total subset for validation
    
    BATCH_SIZE     = 32
    EPOCHS         = 15
    LR             = 1e-4
    WEIGHT_DECAY   = 1e-2
    DROPOUT        = 0.5
    IMG_SIZE       = 224
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()

# ==========================================
# 2. DATASET LOADER (Fetches the exact subset)
# ==========================================
class SentinelSubsetDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_subset_samples(root_dir, max_samples):
    all_samples = []
    for idx, class_name in enumerate(cfg.CLASSES):
        s2_path = os.path.join(root_dir, class_name, 's2')
        if os.path.exists(s2_path):
            imgs = [os.path.join(s2_path, f) for f in os.listdir(s2_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
            for img_p in imgs:
                all_samples.append((img_p, idx))
    
    # Shuffle and pick the exact subset size you use for CycleGAN
    random.shuffle(all_samples)
    return all_samples[:max_samples]

# Prepare Subsets
train_samples = get_subset_samples(cfg.DATA_ROOT, cfg.MAX_TRAIN_SAMPLES)
val_samples   = get_subset_samples(cfg.DATA_ROOT, cfg.MAX_VAL_SAMPLES)

# ==========================================
# 3. TRANSFORMATIONS & LOADERS
# ==========================================
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
train_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

val_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

train_ds = SentinelSubsetDataset(train_samples, transform=train_transform)
val_ds   = SentinelSubsetDataset(val_samples, transform=val_transform)

train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

# ==========================================
# 4. MODEL (ResNet-18)
# ==========================================
model = models.resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(cfg.DROPOUT),
    nn.Linear(256, 4)
)
model = model.to(cfg.DEVICE)

optimizer = optim.AdamW(model.fc.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 5. TRAINING
# ==========================================
print(f"Training on subset of {len(train_samples)} images...")
for epoch in range(cfg.EPOCHS):
    model.train()
    for images, labels in train_dl:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    
    # Basic Val Check
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_dl:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            correct += (model(images).argmax(1) == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Val Acc: {100*correct/len(val_samples):.2f}%")

torch.save(model.state_dict(), 'subset_optical_classifier.pth')