# KITTI Dataset Integration Guide for PointPillars

## Table of Contents
1. [Overview](#overview)
2. [Downloading KITTI Dataset](#downloading-kitti-dataset)
3. [Dataset Structure](#dataset-structure)
4. [Quick Start](#quick-start)
5. [Detailed Setup](#detailed-setup)
6. [Training](#training)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The **KITTI 3D Object Detection Benchmark** is an excellent choice for training PointPillars:

✅ **Pros:**
- ~7,481 labeled training samples with high-quality annotations
- Real-world LiDAR data from Velodyne HDL-64E
- Compatible format with PointPillars (x, y, z, intensity)
- Three main classes: Car, Pedestrian, Cyclist
- Widely used benchmark with good documentation
- **Free to download** for research/educational use

⚠️ **Considerations:**
- Large dataset (~29 GB for point clouds)
- Requires preprocessing to convert to PointPillars format
- Real-world data (different from CARLA but useful for pre-training)

---

## Downloading KITTI Dataset

### Official Download Links

Visit: **https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d**

### Required Downloads:

| File | Size | Description | Required |
|------|------|-------------|----------|
| **data_object_velodyne.zip** | ~29 GB | LiDAR point clouds (.bin files) | ✅ Yes |
| **data_object_label_2.zip** | ~5 MB | 3D bounding box labels (.txt files) | ✅ Yes |
| **data_object_calib.zip** | ~16 MB | Camera-LiDAR calibration files | ⚠️ Optional* |

*Calibration files are needed if you want to use camera data or accurate coordinate transformations. For basic PointPillars training, they're optional.

### Download Steps:

1. **Register/Login** at KITTI website (free account)
2. **Download the three files** above
3. **Extract to a directory**, e.g., `YOUR_DATA_SET_DIRECTORY/KITTI/`

### Alternative: Partial Download (For Testing)

If you want to test first before downloading 29 GB:

```bash
# You can download just a subset using wget with byte ranges
# Or use the KITTI raw data which has smaller chunks
# But for best results, use the full object detection dataset
```

---

## Dataset Structure

After extraction, your directory should look like this:

```
YOUR_DATA_SET_DIRECTORY/KITTI/
├── training/
│   ├── calib/              # 7,481 calibration files
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   └── ...
│   ├── image_2/            # 7,481 left camera images (PNG)
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── label_2/            # 7,481 label files (TXT)
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   └── ...
│   └── velodyne/           # 7,481 point cloud files (BIN)
│       ├── 000000.bin
│       ├── 000001.bin
│       └── ...
└── testing/
    ├── calib/
    ├── image_2/
    └── velodyne/           # No labels (for competition submission)
```

### File Formats:

#### 1. Velodyne Point Clouds (`.bin`)
- Binary file with `float32` values
- Each point: `[x, y, z, intensity]` (4 floats = 16 bytes)
- Coordinate system: Velodyne sensor frame
  - X-axis: forward
  - Y-axis: left
  - Z-axis: up

#### 2. Labels (`.txt`)
Each line contains one 3D object with 15 values:
```
type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom height width length x y z rotation_y [score]
```

**Important fields for PointPillars:**
- `type`: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare
- `height, width, length`: Object dimensions in meters (camera frame)
- `x, y, z`: 3D object location in camera coordinates (meters)
- `rotation_y`: Rotation around Y-axis in camera coordinates

---

## Quick Start

### Step 1: Download KITTI Dataset

```bash
# Create directory
mkdir -p YOUR_DATA_SET_DIRECTORY/KITTI
cd YOUR_DATA_SET_DIRECTORY/KITTI

# Download files from KITTI website
# (Manual download - see links above)

# Extract
unzip data_object_velodyne.zip
unzip data_object_label_2.zip
unzip data_object_calib.zip  # Optional
```

### Step 2: Preprocess KITTI Data

```bash
# Activate ME455_TeamC environment
conda activate tfuse

# Navigate to PointPillars directory
cd /path/to/PointPillars-LiDAR

# Edit preprocessing script to set your KITTI path
# Open preprocessing/kitti_preprocessing.py
# Change line: KITTI_ROOT = 'YOUR_DATA_SET_DIRECTORY/KITTI'

# Run preprocessing
python preprocessing/kitti_preprocessing.py
```

**This will:**
- Load all KITTI velodyne point clouds (`.bin` files)
- Load all 3D bounding box labels (`.txt` files)
- Convert coordinate systems (Camera → Velodyne)
- Filter objects by class (Cars, Pedestrians)
- Save to PointPillars format: `processed_point_clouds.pkl`

**Expected output:**
```
Loading KITTI dataset from: YOUR_DATA_SET_DIRECTORY/KITTI
Split: training
Found 7481 samples in training set
Processing 7481 samples...
Processing KITTI samples: 100%|██████████| 7481/7481 [02:15<00:00, 55.12it/s]

Processed 7481 samples with labels
Saved processed data to: YOUR_DATA_SET_DIRECTORY/KITTI/processed_data/training/processed_point_clouds.pkl

Dataset Statistics:
  Total samples: 7481
  Total objects: 28742
  Avg objects per sample: 3.84

Class Distribution:
  Vehicle: 23573
  Pedestrian: 4487
  Cyclist: 682
```

### Step 3: Train PointPillars

```bash
# Edit training script paths
# Open training/train_pointpillars_kitti.py
# Change:
#   DATA_PATH = 'YOUR_DATA_SET_DIRECTORY/KITTI/processed_data/training/processed_point_clouds.pkl'
#   MODEL_SAVE_PATH = 'YOUR_DATA_SET_DIRECTORY/KITTI/models/pointpillars_kitti.pth'

# Start training
python training/train_pointpillars_kitti.py
```

### Step 4: Test Trained Model

```bash
# Use the trained model with manual_control.py
# (You'll need to modify manual_control.py to load the KITTI-trained model)
python manual_control.py
```

---

## Detailed Setup

### Preprocessing Options

The preprocessing script has several configuration options:

```python
preprocess_kitti_dataset(
    kitti_root='YOUR_DATA_SET_DIRECTORY/KITTI',
    output_file='output/path.pkl',
    split='training',           # 'training' or 'testing'
    max_samples=None,          # None = all samples, or set a number like 1000
    filter_classes=[1, 2]      # 1=Vehicle, 2=Pedestrian, 3=Cyclist, None=all
)
```

#### Filter by Class

To train only on specific object types:

```python
# Only vehicles
filter_classes=[1]

# Vehicles and pedestrians
filter_classes=[1, 2]

# All classes
filter_classes=None
```

#### Subsample Dataset (For Testing)

To quickly test with a smaller dataset:

```python
# Use only first 1000 samples
max_samples=1000
```

### Coordinate System Conversion

KITTI has two coordinate systems:

**Camera Coordinate System** (labels are in this frame):
- X-axis: right
- Y-axis: down
- Z-axis: forward

**Velodyne Coordinate System** (point clouds are in this frame):
- X-axis: forward
- Y-axis: left
- Z-axis: up

**The preprocessing script automatically converts** camera coordinates to velodyne coordinates:

```python
# Transformation applied:
x_velodyne = z_camera
y_velodyne = -x_camera
z_velodyne = -y_camera + height/2
```

---

## Training

### Training Configuration

The KITTI training script has optimized hyperparameters:

```python
BATCH_SIZE = 4              # Adjust based on your GPU memory
NUM_EPOCHS = 50             # KITTI typically needs 50-100 epochs
LEARNING_RATE = 0.001       # Initial learning rate
WEIGHT_DECAY = 1e-4         # L2 regularization
```

### GPU Memory Requirements

| Batch Size | GPU Memory | Training Speed |
|------------|------------|----------------|
| 1 | ~2 GB | Slow |
| 2 | ~4 GB | Medium |
| 4 | ~6 GB | Fast |
| 8 | ~12 GB | Very Fast |

If you run out of GPU memory, reduce `BATCH_SIZE`.

### Expected Training Time

| Hardware | Time per Epoch | Total (50 epochs) |
|----------|----------------|-------------------|
| CPU only | ~2-3 hours | 4-6 days |
| GTX 1060 | ~10 minutes | ~8 hours |
| RTX 2080 | ~5 minutes | ~4 hours |
| RTX 3090 | ~3 minutes | ~2.5 hours |

### Monitoring Training

The training script will output:

```
Epoch 1/50 - Average Loss: 0.1234
  Learning rate: 0.001000
  Checkpoint saved: model_epoch_10.pth

Epoch 15/50 - Average Loss: 0.0856
  Learning rate: 0.000500  <- LR reduced
  New best model saved: model_best.pth (loss: 0.0856)
```

### Saved Model Files

After training:

```
YOUR_DATA_SET_DIRECTORY/KITTI/models/
├── pointpillars_kitti.pth              # Final model
├── pointpillars_kitti_best.pth         # Best model (lowest loss)
├── pointpillars_kitti_epoch_10.pth     # Checkpoint at epoch 10
├── pointpillars_kitti_epoch_20.pth     # Checkpoint at epoch 20
└── ...
```

---

## Using KITTI-Trained Model with CARLA

### Domain Gap Consideration

**Important:** KITTI is real-world data, CARLA is synthetic. There's a **domain gap**.

**Options:**

1. **Pre-training Strategy** (Recommended):
   - Pre-train on KITTI (real-world features)
   - Fine-tune on CARLA data (adapt to synthetic)

2. **Direct Use**:
   - Use KITTI model directly in CARLA
   - May have lower accuracy but useful as baseline

### Fine-tuning on CARLA

```python
# 1. Load KITTI pre-trained model
model = PointPillarsKITTI()
model.load_state_dict(torch.load('kitti_trained_model.pth'))

# 2. Collect small CARLA dataset (~500 samples)
# (Use methods from DATASET_AND_INTEGRATION_GUIDE.md)

# 3. Fine-tune on CARLA data
# - Use lower learning rate (1e-4 or 1e-5)
# - Train for fewer epochs (10-20)
# - This adapts model to CARLA's synthetic characteristics
```

---

## Troubleshooting

### Issue: "FileNotFoundError: Velodyne directory not found"

**Cause:** KITTI dataset not extracted properly

**Solution:**
```bash
# Check directory structure
ls YOUR_DATA_SET_DIRECTORY/KITTI/training/

# Should show: calib/  image_2/  label_2/  velodyne/

# If missing, extract again:
cd YOUR_DATA_SET_DIRECTORY/KITTI
unzip -o data_object_velodyne.zip
```

### Issue: "Out of memory" during training

**Cause:** Batch size too large for your GPU

**Solution:**
```python
# In train_pointpillars_kitti.py, reduce batch size:
BATCH_SIZE = 2  # or even 1
```

### Issue: Training loss not decreasing

**Possible causes:**

1. **Learning rate too high/low**
   ```python
   LEARNING_RATE = 0.0005  # Try reducing
   ```

2. **Not enough epochs**
   ```python
   NUM_EPOCHS = 100  # KITTI may need more epochs
   ```

3. **Data preprocessing issue**
   - Verify processed data: `python preprocessing/kitti_preprocessing.py`
   - Check that labels are loaded correctly
   - Ensure coordinate transformation is correct

### Issue: "Coordinate mismatch" or poor detection

**Cause:** Camera-to-Velodyne transformation may need tuning

**Solution:** The preprocessing script uses standard KITTI transformation. If results are poor, verify:

```python
# In kitti_preprocessing.py, check the transformation:
x_vel = z_cam
y_vel = -x_cam
z_vel = -y_cam + height/2

# This is standard for KITTI, but double-check your results
```

### Issue: "Label format error"

**Cause:** Corrupted or non-standard KITTI labels

**Solution:**
```python
# Add error handling in preprocessing
try:
    points, labels = dataset[idx]
except Exception as e:
    print(f"Error at index {idx}: {e}")
    continue
```

---

## Verification Checklist

Before training, verify:

- [ ] KITTI dataset downloaded (velodyne + labels)
- [ ] Files extracted to correct directory structure
- [ ] Preprocessing script runs without errors
- [ ] Processed pickle file created successfully
- [ ] Verification output shows correct number of samples (7481)
- [ ] Class distribution looks reasonable (Cars >> Pedestrians >> Cyclists)
- [ ] Training script paths updated correctly
- [ ] ME455_TeamC environment activated (`conda activate tfuse`)
- [ ] PyTorch 1.11.0 confirmed: `python -c "import torch; print(torch.__version__)"`

---

## Advanced: Custom KITTI Splits

KITTI doesn't provide official train/val split. Create your own:

```python
# In kitti_preprocessing.py

# Split 1: First 6000 for training, rest for validation
def create_train_val_split(kitti_root, train_samples=6000):
    dataset = KITTIDataset(kitti_root, 'training')

    # Training split
    preprocess_kitti_dataset(
        kitti_root=kitti_root,
        output_file='processed_data/train/processed_point_clouds.pkl',
        split='training',
        max_samples=train_samples
    )

    # Validation split (skip first train_samples)
    # You'd need to modify the script to support offset
```

Or use community splits like **train.txt** and **val.txt** from 3D object detection papers.

---

## Summary

### Workflow:

1. ✅ **Download KITTI** (~29 GB velodyne + ~5 MB labels)
2. ✅ **Run preprocessing** script: `python preprocessing/kitti_preprocessing.py`
3. ✅ **Train model**: `python training/train_pointpillars_kitti.py`
4. ✅ **Evaluate** on CARLA or collect CARLA data for fine-tuning

### Key Benefits of KITTI:

- ✅ Large-scale dataset with 7,481 labeled samples
- ✅ Real-world LiDAR data
- ✅ High-quality annotations
- ✅ Widely used benchmark (easy to compare results)
- ✅ Good for pre-training before CARLA fine-tuning

### Files Created:

- ✅ [preprocessing/kitti_preprocessing.py](preprocessing/kitti_preprocessing.py) - Full preprocessing pipeline
- ✅ [training/train_pointpillars_kitti.py](training/train_pointpillars_kitti.py) - KITTI-optimized training
- ✅ This guide - Complete documentation

You're ready to train on KITTI! 🚀

---

## Additional Resources

- **KITTI Website:** https://www.cvlibs.net/datasets/kitti/
- **KITTI Paper:** "Vision meets Robotics: The KITTI Dataset" (Geiger et al., 2013)
- **PointPillars Paper:** https://arxiv.org/abs/1812.05784
- **KITTI Devkit:** https://github.com/KITTI-dev-kit (for advanced evaluation)
