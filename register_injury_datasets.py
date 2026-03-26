#!/usr/bin/env python3

"""
Injury Dataset Registration for Mask2Former Instance Segmentation

This script registers datasets for injury segmentation training:
- Single stage training with manual annotations only

Expected directory structure:
dataset_root/
  ├── images/
  └── annotations/
      ├── train_annotations.json
      └── test_annotations.json

For cropped evaluation (optional):
dataset_root/
  ├── images/
  ├── images_test_cropped/
  └── annotations/
      ├── train_annotations.json
      ├── test_annotations.json
      └── test_cropped_annotations.json
"""

import os
import json
from detectron2.data.datasets import register_coco_instances


def _register_injury_datasets(annotations_dir, images_dir, dataset_root):
    """Register injury instance segmentation datasets."""
    registered = []

    # Training dataset
    print(f"   📊 Training Dataset")
    train_ann = os.path.join(annotations_dir, "train_annotations.json")

    if os.path.exists(train_ann):
        register_coco_instances("injury_train", {}, train_ann, images_dir)
        registered.append("injury_train")
        print(f"      ✅ Registered injury_train")

        # Count images and annotations
        with open(train_ann, 'r') as f:
            data = json.load(f)
        print(f"      📈 Training images: {len(data['images'])}")
        print(f"      📈 Training annotations: {len(data['annotations'])}")
    else:
        print(f"      ❌ Train annotations not found: {train_ann}")

    # Test/Validation dataset
    print(f"   🎯 Test Dataset")
    test_ann = os.path.join(annotations_dir, "test_annotations.json")

    if os.path.exists(test_ann):
        register_coco_instances("injury_val", {}, test_ann, images_dir)
        registered.append("injury_val")
        print(f"      ✅ Registered injury_val")

        with open(test_ann, 'r') as f:
            data = json.load(f)
        print(f"      📈 Test images: {len(data['images'])}")
        print(f"      📈 Test annotations: {len(data['annotations'])}")
    else:
        print(f"      ⚠️  Test annotations not found, using train for validation")
        if os.path.exists(train_ann):
            register_coco_instances("injury_val", {}, train_ann, images_dir)
            registered.append("injury_val")

    # Cropped test set (4x quadrants for evaluation) - optional
    cropped_ann = os.path.join(annotations_dir, "test_cropped_annotations.json")
    cropped_images_dir = os.path.join(dataset_root, "images_test_cropped")

    if os.path.exists(cropped_ann) and os.path.exists(cropped_images_dir):
        register_coco_instances("injury_val_cropped", {}, cropped_ann, cropped_images_dir)
        registered.append("injury_val_cropped")
        print(f"      ✅ Registered injury_val_cropped (quadrant crops)")

        with open(cropped_ann, 'r') as f:
            data = json.load(f)
        print(f"      📈 Cropped test images: {len(data['images'])} (4x quadrants)")
        print(f"      📈 Cropped test annotations: {len(data['annotations'])}")

    # Full training set with crops (optional, for production training)
    train_full_ann = os.path.join(annotations_dir, "train_full_annotations.json")
    train_cropped_ann = os.path.join(annotations_dir, "train_cropped_annotations.json")
    train_cropped_images_dir = os.path.join(dataset_root, "images_train_cropped")

    if os.path.exists(train_full_ann):
        register_coco_instances("injury_train_full", {}, train_full_ann, images_dir)
        registered.append("injury_train_full")
        print(f"      ✅ Registered injury_train_full")

    if os.path.exists(train_cropped_ann) and os.path.exists(train_cropped_images_dir):
        register_coco_instances("injury_train_cropped", {}, train_cropped_ann, train_cropped_images_dir)
        registered.append("injury_train_cropped")
        print(f"      ✅ Registered injury_train_cropped")

        # Also register as val for monitoring
        register_coco_instances("injury_val_train_cropped", {}, train_cropped_ann, train_cropped_images_dir)
        registered.append("injury_val_train_cropped")
        print(f"      ✅ Registered injury_val_train_cropped (for monitoring)")

    return registered


def register_injury_datasets(
    dataset_root: str = "injury_dataset",
):
    """
    Register datasets for injury instance segmentation.

    Expected structure:
    dataset_root/
    ├── images/
    └── annotations/
        ├── train_annotations.json
        └── test_annotations.json

    Args:
        dataset_root: Path to dataset directory
    """

    print("🔄 Registering injury segmentation datasets...")
    print(f"   Dataset root: {dataset_root}")

    # Check if dataset root exists
    if not os.path.exists(dataset_root):
        print(f"⚠️  Warning: Dataset not found at {dataset_root}")
        print(f"   Please create the dataset structure")
        return []

    images_dir = os.path.join(dataset_root, "images")
    annotations_dir = os.path.join(dataset_root, "annotations")

    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return []

    if not os.path.exists(annotations_dir):
        print(f"❌ Annotations directory not found: {annotations_dir}")
        return []

    registered_datasets = _register_injury_datasets(annotations_dir, images_dir, dataset_root)

    print(f"\n✅ Injury dataset registration completed!")
    print(f"   📊 Datasets registered: {len(registered_datasets)}")
    print(f"   Classes: ['injury']")

    return registered_datasets


def register_all_injury(_root="injury_dataset"):
    """Main registration function to be called from other scripts."""
    return register_injury_datasets(dataset_root=_root)


def check_dataset_structure(dataset_root):
    """Check dataset structure and files."""
    print("🔍 Checking injury dataset structure...")

    issues = []

    if not os.path.exists(dataset_root):
        issues.append(f"Dataset root not found: {dataset_root}")
        return issues

    images_dir = os.path.join(dataset_root, "images")
    annotations_dir = os.path.join(dataset_root, "annotations")

    if not os.path.exists(images_dir):
        issues.append(f"Missing images directory: {images_dir}")
    else:
        image_count = len([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        print(f"   ✅ Images directory found: {image_count} images")

    if not os.path.exists(annotations_dir):
        issues.append(f"Missing annotations directory: {annotations_dir}")
    else:
        print(f"   ✅ Annotations directory found")

        required_files = ["train_annotations.json"]
        optional_files = ["test_annotations.json", "test_cropped_annotations.json"]

        for file in required_files:
            file_path = os.path.join(annotations_dir, file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"   ✅ Found: {file} ({len(data['images'])} images, {len(data['annotations'])} annotations)")
            else:
                issues.append(f"Missing required: {file}")

        for file in optional_files:
            file_path = os.path.join(annotations_dir, file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"   ✅ Found: {file} ({len(data['images'])} images, {len(data['annotations'])} annotations)")
            else:
                print(f"   ℹ️  Optional: {file} not found")

    if issues:
        print("\n❌ Dataset issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print(f"\n💡 Expected structure:")
        print(f"   {dataset_root}/")
        print(f"   ├── images/")
        print(f"   └── annotations/")
        print(f"       ├── train_annotations.json  (required)")
        print(f"       └── test_annotations.json   (optional)")

    return issues


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register injury segmentation datasets")
    parser.add_argument("--dataset_root", default="injury_dataset",
                       help="Root directory of injury dataset")
    parser.add_argument("--check", action="store_true",
                       help="Check dataset structure without registering")

    args = parser.parse_args()

    # Check structure
    issues = check_dataset_structure(args.dataset_root)

    if not args.check:
        # Register datasets
        register_injury_datasets(dataset_root=args.dataset_root)
