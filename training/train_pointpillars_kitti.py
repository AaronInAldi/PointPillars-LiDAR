"""
PointPillars Training Script for KITTI Dataset
Compatible with Python 3.7.13 and PyTorch 1.11.0

This is a specialized version for KITTI dataset with optimized hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # Modified for PyTorch 1.11.0 compatibility
import os


class PointPillarsKITTI(nn.Module):
    """PointPillars model optimized for KITTI dataset"""

    def __init__(self, num_classes=3):
        """
        Args:
            num_classes: Number of object classes
                        1 = Vehicle, 2 = Pedestrian, 3 = Cyclist
        """
        super(PointPillarsKITTI, self).__init__()

        # Pillar Feature Net
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Detection head
        self.fc1 = nn.Linear(128 * 1000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)  # [conf, class, x, y, z, w, h, l]

        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pillar feature extraction
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Detection head
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x


class KITTIPointCloudDataset(Dataset):
    """Dataset loader for preprocessed KITTI data"""

    def __init__(self, pickle_file, max_points=1000):
        """
        Args:
            pickle_file: Path to preprocessed KITTI pickle file
            max_points: Maximum number of points per sample
        """
        print(f"Loading KITTI dataset from: {pickle_file}")

        with open(pickle_file, 'rb') as f:
            self.all_point_clouds, self.all_labels = pickle.load(f)

        self.max_points = max_points

        print(f"Loaded {len(self.all_point_clouds)} samples")

        # Calculate max label size
        self.max_labels = max(
            len(labels) if len(labels) > 0 else 1
            for labels in self.all_labels
        )

        print(f"Max points per sample: {self.max_points}")
        print(f"Max labels per sample: {self.max_labels}")

    def __len__(self):
        return len(self.all_point_clouds)

    def __getitem__(self, idx):
        pc = self.all_point_clouds[idx]
        labels = self.all_labels[idx]

        # Downsample or pad point cloud
        if len(pc) > self.max_points:
            # Random sampling
            indices = np.random.choice(len(pc), self.max_points, replace=False)
            pc = pc[indices]
        else:
            # Pad with zeros
            padding = np.zeros((self.max_points - len(pc), 4), dtype=np.float32)
            pc = np.vstack([pc, padding])

        # Convert labels to fixed-size array
        # Format: [class_id, center_x, center_y, center_z, size_x, size_y, size_z]
        label_array = np.zeros((self.max_labels, 7), dtype=np.float32)

        for i, (class_id, bbox) in enumerate(labels):
            if i >= self.max_labels:
                break
            label_array[i, 0] = class_id
            label_array[i, 1:7] = bbox

        # Reshape point cloud for Conv2D: (4, max_points, 1)
        pc = pc.transpose(1, 0).reshape(4, self.max_points, 1)

        return (
            torch.tensor(pc, dtype=torch.float32),
            torch.tensor(label_array, dtype=torch.float32)
        )


def create_kitti_data_loader(pickle_file, batch_size=4, shuffle=True):
    """
    Create DataLoader for KITTI dataset

    Args:
        pickle_file: Path to preprocessed KITTI data
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
    """
    dataset = KITTIPointCloudDataset(pickle_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_model_kitti(model, data_loader, criterion, optimizer, scheduler=None,
                     num_epochs=50, save_path='model.pth', save_interval=10,
                     device=None):
    """
    Training loop for PointPillars on KITTI

    Args:
        model: PointPillars model
        data_loader: KITTI DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of training epochs
        save_path: Path to save model checkpoints
        save_interval: Save checkpoint every N epochs
        device: Device to train on (None = auto-detect)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nTraining on device: {device}")
    model.to(device)

    scaler = GradScaler()  # AMP for faster training

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{running_loss / num_batches:.4f}'
            })

        # Epoch statistics
        epoch_loss = running_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {epoch_loss:.4f}")

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'{save_path}_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = save_path.replace('.pth', '_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, best_model_path)
            print(f"  New best model saved: {best_model_path} (loss: {best_loss:.4f})")

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model saved: {save_path}")


if __name__ == '__main__':
    # ========================================================================
    # Configuration
    # ========================================================================

    # Dataset paths
    DATA_PATH = 'YOUR_DATA_SET_DIRECTORY/KITTI/processed_data/training/processed_point_clouds.pkl'
    MODEL_SAVE_PATH = 'YOUR_DATA_SET_DIRECTORY/KITTI/models/pointpillars_kitti.pth'

    # Training hyperparameters (optimized for KITTI)
    BATCH_SIZE = 4          # Reduce if running out of memory
    NUM_EPOCHS = 50         # KITTI typically needs 50-100 epochs
    LEARNING_RATE = 0.001   # Higher than original for faster convergence
    WEIGHT_DECAY = 1e-4     # L2 regularization

    # ========================================================================
    # Setup
    # ========================================================================

    print("="*70)
    print("PointPillars Training on KITTI Dataset")
    print("="*70)
    print(f"Configuration:")
    print(f"  Data path: {DATA_PATH}")
    print(f"  Model save path: {MODEL_SAVE_PATH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print("="*70)

    # Create model save directory
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # ========================================================================
    # Model, Loss, Optimizer
    # ========================================================================

    # Initialize model
    model = PointPillarsKITTI(num_classes=3)
    print(f"\nModel architecture:")
    print(model)

    # Loss function (MSE for bounding box regression)
    criterion = nn.MSELoss()

    # Optimizer (Adam with weight decay)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler (reduce LR on plateau)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,  # Reduce LR every 15 epochs
        gamma=0.5      # Multiply LR by 0.5
    )

    # ========================================================================
    # Data Loading
    # ========================================================================

    print("\nLoading KITTI dataset...")
    data_loader = create_kitti_data_loader(
        DATA_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(f"Number of batches: {len(data_loader)}")

    # ========================================================================
    # Training
    # ========================================================================

    print("\nStarting training...")
    train_model_kitti(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        save_path=MODEL_SAVE_PATH,
        save_interval=10
    )

    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    print(f"Final model saved at: {MODEL_SAVE_PATH}")
    print(f"Best model saved at: {MODEL_SAVE_PATH.replace('.pth', '_best.pth')}")
