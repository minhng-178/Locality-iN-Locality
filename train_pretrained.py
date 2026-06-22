"""
Training script for LNL_Ti on GTSRB.
Run on Colab with GPU (T4): upload this file + LNL.py + models/ folder.
This generates LNL_Ti_GTSRB_pretrained.pt for plug-and-play with Instructions.ipynb.
"""
import os, sys, gc, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Ensure the local module path is correct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from LNL import LNL_Ti as create_model

# ─── Config ───────────────────────────────────────────────────
BATCH_SIZE  = 64
NUM_EPOCHS  = 15
LR          = 5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
MIXUP_ALPHA = 0.2
MIXUP_PROB  = 0.5
LABEL_SMOOTHING = 0.1
DROP_PATH_RATE  = 0.1
NUM_WORKERS = 2
IMG_SIZE    = 224
NUM_CLASSES = 43

GTSRB_MEAN = [0.3337, 0.3064, 0.3171]
GTSRB_STD  = [0.2672, 0.2564, 0.2629]

CACHE_DIR = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ─── Use TF32 if available (Ampere+) ───────────────────────────
if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')

# ─── CachedDataset ─────────────────────────────────────────────
def cache_dataset(root, cache_file):
    mmap_path = cache_file + '.images.npy'
    if os.path.exists(mmap_path):
        mmap = np.memmap(mmap_path, dtype=np.uint8, mode='r',
                         shape=(os.path.getsize(mmap_path) // (3 * IMG_SIZE * IMG_SIZE),
                                3, IMG_SIZE, IMG_SIZE))
        print(f'Found cache: {mmap_path} ({mmap.shape[0]} images)')
        del mmap
        return

    dataset = dsets.ImageFolder(root, transform=transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]))
    N = len(dataset)
    mmap = np.memmap(mmap_path, dtype=np.uint8, mode='w+', shape=(N, 3, IMG_SIZE, IMG_SIZE))
    labels_list = []
    for i, (img, label) in enumerate(dataset):
        arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
        mmap[i] = arr
        labels_list.append(label)
        if (i + 1) % 5000 == 0:
            print(f'  Cached {i+1}/{N}')
    del mmap
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    torch.save({'labels': labels_tensor, 'classes': dataset.classes,
                'shape': (N, 3, IMG_SIZE, IMG_SIZE)}, cache_file)
    print(f'Saved: {cache_file} + {mmap_path}')

class CachedDataset(Dataset):
    def __init__(self, cache_file, transform=None):
        data = torch.load(cache_file, weights_only=True)
        self.labels = data['labels']
        self.classes = data.get('classes', [str(i) for i in range(max(self.labels).item() + 1)])
        self.targets = self.labels.tolist()
        self.transform = transform
        mmap_path = cache_file + '.images.npy'
        N, C, H, W = data['shape']
        self._mmap = np.memmap(mmap_path, dtype=np.uint8, mode='r', shape=(N, C, H, W))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self._mmap[idx].copy()).float().div(255.0)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ─── Mixup ─────────────────────────────────────────────────────
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ─── Main ──────────────────────────────────────────────────────
def main():
    # Cache datasets
    print('Caching training set...')
    cache_dataset('./data/GTSRB/Final_Training/Images', f'{CACHE_DIR}/gtsrb_train_224.pt')
    gc.collect()

    print('Caching test set...')
    cache_dataset('./data/GTSRB/test', f'{CACHE_DIR}/gtsrb_test_224.pt')
    gc.collect()

    # Transforms — NO Normalize: model handles normalization in forward_features
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomErasing(p=0.25),
    ])
    test_transform = None  # model normalizes internally

    # Datasets
    trainset = CachedDataset(f'{CACHE_DIR}/gtsrb_train_224.pt', transform=train_transform)
    testset  = CachedDataset(f'{CACHE_DIR}/gtsrb_test_224.pt',  transform=test_transform)

    # WeightedRandomSampler for class imbalance
    train_root = './data/GTSRB/Final_Training/Images'
    class_list = sorted(os.listdir(train_root))
    class_counts = [len(os.listdir(os.path.join(train_root, c))) for c in class_list]
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[trainset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0),
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f'Train: {len(trainset)} samples, {len(train_loader)} batches')
    print(f'Test:  {len(testset)} samples, {len(test_loader)} batches')

    # Model
    model = create_model(pretrained=False, num_classes=NUM_CLASSES, drop_path_rate=DROP_PATH_RATE)
    model = model.to(device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer + Scheduler
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    # AMP
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Training
    best_acc = 0.0
    ckpt_path = 'LNL_Ti_GTSRB_best.pt'

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if np.random.random() < MIXUP_PROB:
                images, Y_a, Y_b, lam = mixup_data(images, labels)
            else:
                Y_a, Y_b, lam = labels, labels, 1.0

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred = model(images)
                    cost = mixup_criterion(loss_fn, pred, Y_a, Y_b, lam)
                scaler.scale(cost).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(images)
                cost = mixup_criterion(loss_fn, pred, Y_a, Y_b, lam)
                cost.backward()
                optimizer.step()

            running_loss += cost.item()

            if (i + 1) % 200 == 0:
                avg_loss = running_loss / (i + 1)
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Iter [{i+1}/{len(train_loader)}], '
                      f'Loss: {avg_loss:.4f}, LR: {lr_now:.2e}, Time: {elapsed:.0f}s')

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - t0

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for val_images, val_labels in test_loader:
                val_images = val_images.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)
                outputs = model(val_images)
                _, predicted = torch.max(outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
        val_acc = 100.0 * correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_acc': best_acc,
            }, ckpt_path)
            print(f'*** Best model saved (acc={best_acc:.2f}%) ***')

        print(f'=== Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | '
              f'Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}% | Time: {epoch_time:.0f}s ===')

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f'\nDone! Best accuracy: {best_acc:.2f}%')
    print(f'Checkpoint saved to: {ckpt_path}')

if __name__ == '__main__':
    main()
