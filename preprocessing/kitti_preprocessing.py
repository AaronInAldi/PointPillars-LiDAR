"""
KITTI Dataset Preprocessing for PointPillars Training
Compatible with Python 3.7 and PyTorch 1.11.0

This script converts KITTI 3D Object Detection dataset to PointPillars format.

KITTI Dataset Download:
- Object Detection Benchmark: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
- Download:
  1. Velodyne point clouds (29 GB): data_object_velodyne.zip
  2. Training labels (5 MB): data_object_label_2.zip
  3. Camera calibration (16 MB): data_object_calib.zip (optional but recommended)

Dataset Structure:
KITTI/
├── training/
│   ├── velodyne/   # LiDAR point clouds (.bin files)
│   ├── label_2/    # 3D bounding box labels (.txt files)
│   └── calib/      # Calibration files (.txt files)
└── testing/
    └── velodyne/
"""

import numpy as np
import pickle
import os
from tqdm import tqdm


class KITTIDataset:
    """KITTI Dataset Loader and Converter"""

    # KITTI class mapping to integer labels
    CLASS_MAP = {
        'Car': 1,
        'Van': 1,
        'Truck': 1,
        'Pedestrian': 2,
        'Person_sitting': 2,
        'Cyclist': 3,
        'Tram': 1,
        'Misc': 0,
        'DontCare': 0
    }

    def __init__(self, kitti_root, split='training'):
        """
        Args:
            kitti_root: Path to KITTI dataset root directory
            split: 'training' or 'testing'
        """
        self.kitti_root = kitti_root
        self.split = split

        self.velodyne_dir = os.path.join(kitti_root, split, 'velodyne')
        self.label_dir = os.path.join(kitti_root, split, 'label_2')
        self.calib_dir = os.path.join(kitti_root, split, 'calib')

        # Get list of sample IDs
        self.sample_ids = self._get_sample_ids()

        print(f"Found {len(self.sample_ids)} samples in {split} set")

    def _get_sample_ids(self):
        """Get list of sample IDs from velodyne directory"""
        if not os.path.exists(self.velodyne_dir):
            raise FileNotFoundError(f"Velodyne directory not found: {self.velodyne_dir}")

        sample_ids = [
            f.split('.')[0]
            for f in os.listdir(self.velodyne_dir)
            if f.endswith('.bin')
        ]
        return sorted(sample_ids)

    def load_velodyne(self, idx):
        """
        Load LiDAR point cloud from .bin file

        Returns:
            points: numpy array of shape (N, 4) with columns [x, y, z, intensity]
        """
        sample_id = self.sample_ids[idx]
        velodyne_file = os.path.join(self.velodyne_dir, f'{sample_id}.bin')

        # KITTI velodyne format: binary file with float32 values
        # Each point: [x, y, z, intensity]
        points = np.fromfile(velodyne_file, dtype=np.float32).reshape(-1, 4)

        return points

    def load_labels(self, idx):
        """
        Load 3D bounding box labels from .txt file

        KITTI label format (15 values per line):
        type truncated occluded alpha bbox_2d(4) dimensions(3) location(3) rotation_y score

        We need:
        - type: object class
        - dimensions: height, width, length (in meters)
        - location: x, y, z in camera coordinates (needs conversion to velodyne)
        - rotation_y: rotation around Y-axis

        Returns:
            labels: list of tuples (class_id, bbox_array)
                    bbox_array: [center_x, center_y, center_z, size_x, size_y, size_z]
        """
        sample_id = self.sample_ids[idx]
        label_file = os.path.join(self.label_dir, f'{sample_id}.txt')

        if not os.path.exists(label_file):
            return []  # No labels for this sample

        labels = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) < 15:
                    continue

                obj_type = parts[0]

                # Skip DontCare regions
                if obj_type == 'DontCare':
                    continue

                # Get class ID
                class_id = self.CLASS_MAP.get(obj_type, 0)
                if class_id == 0:
                    continue  # Skip unknown classes

                # Parse dimensions (height, width, length in meters)
                height = float(parts[8])
                width = float(parts[9])
                length = float(parts[10])

                # Parse location (x, y, z in camera coordinates)
                x_cam = float(parts[11])
                y_cam = float(parts[12])
                z_cam = float(parts[13])

                # Convert from camera coordinates to velodyne coordinates
                # KITTI camera: x=right, y=down, z=forward
                # Velodyne: x=forward, y=left, z=up
                # Transformation: [x_vel, y_vel, z_vel] = [z_cam, -x_cam, -y_cam]
                x_vel = z_cam
                y_vel = -x_cam
                z_vel = -y_cam + height/2  # Adjust for center vs bottom

                # Create bounding box: [center_x, center_y, center_z, size_x, size_y, size_z]
                # In velodyne frame: size_x=length, size_y=width, size_z=height
                bbox = np.array([x_vel, y_vel, z_vel, length, width, height])

                labels.append((class_id, bbox))

        return labels

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """Get a single sample"""
        points = self.load_velodyne(idx)
        labels = self.load_labels(idx)
        return points, labels


def preprocess_kitti_dataset(kitti_root, output_file, split='training',
                             max_samples=None, filter_classes=None):
    """
    Convert KITTI dataset to PointPillars pickle format

    Args:
        kitti_root: Path to KITTI dataset root
        output_file: Path to save processed data (.pkl)
        split: 'training' or 'testing'
        max_samples: Maximum number of samples to process (None = all)
        filter_classes: List of class IDs to keep (None = all)
                       Example: [1, 2] keeps only cars and pedestrians
    """
    print(f"Loading KITTI dataset from: {kitti_root}")
    print(f"Split: {split}")

    # Load dataset
    dataset = KITTIDataset(kitti_root, split)

    # Determine number of samples
    num_samples = len(dataset)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    print(f"Processing {num_samples} samples...")

    all_point_clouds = []
    all_labels = []

    # Process each sample
    for idx in tqdm(range(num_samples), desc="Processing KITTI samples"):
        try:
            points, labels = dataset[idx]

            # Apply point cloud filtering (optional)
            # Remove points too far or too close
            distances = np.sqrt(np.sum(points[:, :3]**2, axis=1))
            mask = (distances > 1.0) & (distances < 70.0)
            points = points[mask]

            # Filter labels by class if specified
            if filter_classes is not None:
                labels = [
                    (class_id, bbox)
                    for class_id, bbox in labels
                    if class_id in filter_classes
                ]

            # Only keep samples with at least one object
            if len(labels) > 0:
                all_point_clouds.append(points)
                all_labels.append(labels)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    print(f"\nProcessed {len(all_point_clouds)} samples with labels")

    # Save to pickle file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump((all_point_clouds, all_labels), f)

    print(f"Saved processed data to: {output_file}")

    # Print statistics
    total_objects = sum(len(labels) for labels in all_labels)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(all_point_clouds)}")
    print(f"  Total objects: {total_objects}")
    print(f"  Avg objects per sample: {total_objects / len(all_point_clouds):.2f}")

    # Class distribution
    class_counts = {}
    for labels in all_labels:
        for class_id, _ in labels:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

    print(f"\nClass Distribution:")
    class_names = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist'}
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names.get(class_id, f'Class_{class_id}')
        print(f"  {class_name}: {count}")


def verify_processed_data(pickle_file):
    """Verify the processed pickle file"""
    print(f"\nVerifying processed data: {pickle_file}")

    with open(pickle_file, 'rb') as f:
        all_point_clouds, all_labels = pickle.load(f)

    print(f"  Number of samples: {len(all_point_clouds)}")
    print(f"  Number of label sets: {len(all_labels)}")

    # Check first sample
    if len(all_point_clouds) > 0:
        pc = all_point_clouds[0]
        labels = all_labels[0]

        print(f"\nFirst sample:")
        print(f"  Point cloud shape: {pc.shape}")
        print(f"  Number of points: {len(pc)}")
        print(f"  Number of objects: {len(labels)}")

        if len(labels) > 0:
            print(f"  First label: class={labels[0][0]}, bbox shape={labels[0][1].shape}")
            print(f"  Bbox values: {labels[0][1]}")


if __name__ == '__main__':
    # Configuration
    KITTI_ROOT = 'YOUR_DATA_SET_DIRECTORY/KITTI'  # Change this to your KITTI path
    OUTPUT_DIR = 'YOUR_DATA_SET_DIRECTORY/KITTI/processed_data'

    # Process training set
    print("="*60)
    print("Processing KITTI Training Set")
    print("="*60)

    preprocess_kitti_dataset(
        kitti_root=KITTI_ROOT,
        output_file=os.path.join(OUTPUT_DIR, 'training', 'processed_point_clouds.pkl'),
        split='training',
        max_samples=None,  # Process all samples, or set a number like 1000
        filter_classes=[1, 2]  # Keep only vehicles (1) and pedestrians (2)
    )

    # Verify the processed data
    verify_processed_data(
        os.path.join(OUTPUT_DIR, 'training', 'processed_point_clouds.pkl')
    )

    print("\n" + "="*60)
    print("KITTI preprocessing complete!")
    print("="*60)
    print(f"\nYou can now train PointPillars with:")
    print(f"  python training/train_pointpillars_training.py")
    print(f"\nMake sure to update the data path in train_pointpillars_training.py to:")
    print(f"  '{os.path.join(OUTPUT_DIR, 'training', 'processed_point_clouds.pkl')}'")
