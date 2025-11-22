# Dataset and Integration Guide for PointPillars with CARLA 0.9.10.1

## Table of Contents
1. [Dataset Options for CARLA Training](#dataset-options)
2. [Generating Your Own CARLA Dataset](#generating-carla-dataset)
3. [Integration with ME455_TeamC](#integration-guide)
4. [Quick Start](#quick-start)

---

## Dataset Options for CARLA Training

### ❌ Original Dataset (Waymo Open Dataset)
The original PointPillars code was designed for the **Waymo Open Dataset**, which:
- Is designed for real-world autonomous driving
- Contains data from real Waymo vehicles with high-end LiDAR sensors
- **NOT suitable for CARLA** because the sensor characteristics and data format differ significantly

### ✅ Recommended: Generate Your Own CARLA Dataset

For vehicle and pedestrian detection in CARLA, you should **generate your own training data** from CARLA simulator.

---

## Generating Your Own CARLA Dataset

### Method 1: Use ME455_TeamC's Autopilot (Recommended)

Your ME455_TeamC project already has a data generation system!

**Location:** `C:\Users\aaron\Projects\CARLA\ME455_TeamC\team_code_autopilot\`

**Steps:**

1. **Start CARLA Server:**
   ```bash
   cd carla
   ./CarlaUE4.sh --world-port=2000 -opengl
   ```

2. **Use the provided datagen script:**
   ```bash
   cd C:\Users\aaron\Projects\CARLA\ME455_TeamC
   ./leaderboard/scripts/datagen.sh <carla root> <working directory>
   ```

3. **Data will be saved in this structure:**
   ```
   - Town
       - Route
           - rgb/           # Camera images
           - depth/         # Depth images
           - semantics/     # Segmentation images
           - lidar/         # 3D point cloud (.npy format)
           - topdown/       # Topdown segmentation maps
           - label_raw/     # 3D bounding boxes for vehicles
           - measurements/  # Ego-agent's position, velocity, metadata
   ```

4. **Convert to PointPillars format:**
   You'll need to write a conversion script that:
   - Reads `.npy` LiDAR files from ME455_TeamC format
   - Reads `label_raw` bounding box data
   - Converts to the pickle format expected by PointPillars training script

   See section below for conversion script template.

---

### Method 2: Create Custom Data Collection Script

Create a standalone CARLA data collection script:

```python
#!/usr/bin/env python3
"""
CARLA LiDAR Data Collection Script for PointPillars Training
Compatible with CARLA 0.9.10.1
"""

import carla
import numpy as np
import pickle
import os
import time

def collect_carla_data(output_dir, num_frames=1000):
    """Collect LiDAR point clouds and bounding boxes from CARLA"""

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Spawn ego vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(True)

    # Spawn LiDAR sensor
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '1000000')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp.set_attribute('range', '100')

    lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    # Storage
    all_point_clouds = []
    all_labels = []

    def lidar_callback(data):
        """Process LiDAR data"""
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = points.reshape(-1, 4)  # [x, y, z, intensity]

        # Get vehicle bounding boxes in the scene
        labels = []
        for actor in world.get_actors().filter('vehicle.*'):
            if actor.id == vehicle.id:
                continue  # Skip ego vehicle

            # Get bounding box
            bb = actor.bounding_box
            transform = actor.get_transform()
            location = transform.location

            # Convert to label format: [type, [center_x, center_y, center_z, size_x, size_y, size_z]]
            label = (
                1,  # vehicle type
                np.array([
                    location.x, location.y, location.z,
                    bb.extent.x * 2, bb.extent.y * 2, bb.extent.z * 2
                ])
            )
            labels.append(label)

        # Also detect pedestrians
        for actor in world.get_actors().filter('walker.pedestrian.*'):
            bb = actor.bounding_box
            transform = actor.get_transform()
            location = transform.location

            label = (
                2,  # pedestrian type
                np.array([
                    location.x, location.y, location.z,
                    bb.extent.x * 2, bb.extent.y * 2, bb.extent.z * 2
                ])
            )
            labels.append(label)

        all_point_clouds.append(points)
        all_labels.append(labels)

        print(f"Collected frame {len(all_point_clouds)}/{num_frames}, "
              f"{len(labels)} objects detected")

    # Attach callback
    lidar.listen(lidar_callback)

    # Collect data
    try:
        while len(all_point_clouds) < num_frames:
            world.tick()
            time.sleep(0.05)
    finally:
        # Cleanup
        lidar.stop()
        lidar.destroy()
        vehicle.destroy()

    # Save data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'processed_point_clouds.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump((all_point_clouds, all_labels), f)

    print(f"Saved {len(all_point_clouds)} frames to {output_file}")

if __name__ == '__main__':
    collect_carla_data('data/carla_training', num_frames=1000)
```

**Usage:**
```bash
# Start CARLA server first
cd carla
./CarlaUE4.sh --world-port=2000 -opengl

# In another terminal, activate ME455_TeamC environment
conda activate tfuse
python carla_data_collection.py
```

---

### Method 3: Use Existing CARLA-Compatible Datasets

#### Option A: CARLA Object Detection Dataset (Community)
- Search for "CARLA object detection dataset" on GitHub
- Look for datasets specifically created for CARLA 0.9.x versions
- Common repositories:
  - `carla-dataset-tools`
  - `CARLA_LiDAR_Dataset`

#### Option B: Synthetic Datasets
- **KITTI Dataset** - Can be adapted but requires format conversion
- **nuScenes** - High-quality but requires extensive preprocessing

---

## Integration with ME455_TeamC

### Step 1: Copy PointPillars Code to ME455_TeamC

```bash
# Navigate to ME455_TeamC
cd C:\Users\aaron\Projects\CARLA\ME455_TeamC

# Create PointPillars directory
mkdir -p team_code_autopilot/perception/pointpillars

# Copy the modified PointPillars files
cp -r /path/to/PointPillars-LiDAR/training team_code_autopilot/perception/pointpillars/
cp /path/to/PointPillars-LiDAR/manual_control.py team_code_autopilot/perception/pointpillars/
```

### Step 2: Verify Environment

```bash
conda activate tfuse
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import carla; print('CARLA imported successfully')"
```

Expected output:
```
PyTorch: 1.11.0
CARLA imported successfully
```

### Step 3: Install Missing Dependencies (if needed)

```bash
conda activate tfuse
pip install pyarrow==6.0.1
```

### Step 4: Integrate with Your Existing Code

You can use PointPillars detection in your ME455_TeamC autopilot:

```python
# In team_code_autopilot/autopilot.py or similar

import sys
sys.path.append('team_code_autopilot/perception/pointpillars/training')
from train_pointpillars_training import PointPillars
import torch

class YourAutopilot:
    def __init__(self):
        # Load PointPillars model
        self.pointpillars = PointPillars()
        self.pointpillars.load_state_dict(
            torch.load('path/to/trained_model.pth')
        )
        self.pointpillars.eval()

    def process_lidar(self, lidar_data):
        # Process LiDAR data (convert to format expected by model)
        # ... your preprocessing code ...

        # Run detection
        with torch.no_grad():
            detections = self.pointpillars(processed_data)

        return detections
```

---

## Quick Start

### For Testing Without Training (Using Pre-existing Models)

If you want to test the PointPillars architecture without training:

1. **Test with manual_control.py:**
   ```bash
   # Start CARLA
   cd carla
   ./CarlaUE4.sh --world-port=2000 -opengl

   # Run manual control (modified version)
   conda activate tfuse
   cd /path/to/PointPillars-LiDAR
   python manual_control.py
   ```

2. **The model will run in inference mode** (even without trained weights)
   - It won't detect objects correctly without training
   - But you can verify the pipeline works

### For Full Training Pipeline

1. **Collect data** (Method 1 or 2 above) → 1000-5000 frames recommended
2. **Preprocess data** → Convert to PointPillars format
3. **Train model:**
   ```bash
   conda activate tfuse
   cd /path/to/PointPillars-LiDAR/training

   # Edit the data path in train_pointpillars_training.py first
   python train_pointpillars_training.py
   ```
4. **Test trained model:**
   ```bash
   python manual_control.py
   ```

---

## Data Format Requirements

### Input Format (for training script)
The training script expects a pickle file containing:
```python
(all_point_clouds, all_labels)
```

Where:
- `all_point_clouds`: List of numpy arrays, each shape `(N, 4)` where N = number of points
  - Columns: `[x, y, z, intensity]`
- `all_labels`: List of tuples `(type, bbox_array)`
  - `type`: Integer (1=vehicle, 2=pedestrian, etc.)
  - `bbox_array`: numpy array `[center_x, center_y, center_z, size_x, size_y, size_z]`

### Conversion from ME455_TeamC Format

```python
import numpy as np
import pickle

def convert_me455_to_pointpillars(me455_data_dir, output_file):
    """Convert ME455_TeamC format to PointPillars format"""
    all_point_clouds = []
    all_labels = []

    # Load .npy files from ME455_TeamC
    lidar_files = sorted(glob.glob(f'{me455_data_dir}/lidar/*.npy'))
    label_files = sorted(glob.glob(f'{me455_data_dir}/label_raw/*.json'))

    for lidar_file, label_file in zip(lidar_files, label_files):
        # Load LiDAR
        points = np.load(lidar_file)  # Should be (N, 4) or (N, 3)
        if points.shape[1] == 3:
            # Add intensity channel if missing
            intensity = np.ones((points.shape[0], 1))
            points = np.hstack([points, intensity])

        # Load labels
        with open(label_file, 'r') as f:
            labels_json = json.load(f)

        labels = []
        for obj in labels_json:
            if obj['class'] == 'vehicle':
                obj_type = 1
            elif obj['class'] == 'pedestrian':
                obj_type = 2
            else:
                continue

            bbox = np.array([
                obj['location']['x'],
                obj['location']['y'],
                obj['location']['z'],
                obj['extent']['x'] * 2,
                obj['extent']['y'] * 2,
                obj['extent']['z'] * 2
            ])
            labels.append((obj_type, bbox))

        all_point_clouds.append(points)
        all_labels.append(labels)

    # Save in PointPillars format
    with open(output_file, 'wb') as f:
        pickle.dump((all_point_clouds, all_labels), f)

    print(f"Converted {len(all_point_clouds)} frames")
```

---

## Troubleshooting

### Issue: "No module named 'carla'"
**Solution:** Add CARLA Python API to your path:
```bash
export PYTHONPATH="${PYTHONPATH}:carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
```

### Issue: "MapLayer not available"
**Solution:** This is expected in CARLA 0.9.10.1. The modified `manual_control.py` handles this gracefully.

### Issue: Training loss not decreasing
**Solution:**
- Collect more training data (minimum 1000 frames)
- Ensure labels are correct
- Check that LiDAR data contains actual objects
- Verify coordinate systems match

---

## Additional Resources

- **CARLA Documentation:** https://carla.readthedocs.io/en/0.9.10/
- **PointPillars Paper:** https://arxiv.org/abs/1812.05784
- **ME455_TeamC README:** See your project's main README for data generation details

---

## Summary

**Recommended Workflow for ME455_TeamC Integration:**

1. ✅ Use ME455_TeamC's existing data generation system
2. ✅ Write a converter script (template provided above)
3. ✅ Train PointPillars model with converted data
4. ✅ Integrate trained model into your autopilot

**Key Benefits:**
- No need for external datasets
- Data perfectly matches your CARLA setup
- Same sensor configurations as your project
- Labels are automatically generated by CARLA
