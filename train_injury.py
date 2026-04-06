#!/usr/bin/env python3

"""
Injury Segmentation Training Script

Single-stage training for injury instance segmentation using manual annotations.
Uses the same Mask2Former architecture and pretrained weights as myotube segmentation.

Usage:
    # Basic training
    python train_injury.py --dataset injury_dataset

    # Custom config
    python train_injury.py --dataset injury_dataset --config custom_config.yaml

    # Resume training
    python train_injury.py --dataset injury_dataset --resume

    # Evaluation only
    python train_injury.py --dataset injury_dataset --eval-only

    # Custom output directory
    python train_injury.py --dataset injury_dataset --output ./my_output
"""

import os
import sys
import argparse
import glob
import json
import random
import shutil
import numpy as np
from PIL import Image

# Import dataset registration
from register_injury_datasets import register_injury_datasets
from detectron2.data.datasets import register_coco_instances

# Import Mask2Former training
from train_net import main as train_main


def split_and_register(dataset_root, num_test, seed=42):
    """Randomly split train_annotations.json into train/test, crop test images into quadrants,
    and register the resulting datasets.

    Returns (train_dataset_name, test_dataset_name, test_images_list) or None on failure.
    """
    annotations_dir = os.path.join(dataset_root, "annotations")
    images_dir = os.path.join(dataset_root, "images")
    train_ann_path = os.path.join(annotations_dir, "train_annotations.json")

    with open(train_ann_path, 'r') as f:
        data = json.load(f)

    all_images = data['images']
    all_annotations = data['annotations']
    categories = data['categories']

    if num_test >= len(all_images):
        print(f"❌ num_test ({num_test}) must be less than total images ({len(all_images)})")
        return None

    # Random split
    rng = random.Random(seed)
    image_indices = list(range(len(all_images)))
    rng.shuffle(image_indices)
    test_indices = set(image_indices[:num_test])

    test_image_ids = set()
    split_train_images = []
    split_test_images = []
    for i, img in enumerate(all_images):
        if i in test_indices:
            split_test_images.append(img)
            test_image_ids.add(img['id'])
        else:
            split_train_images.append(img)

    split_train_anns = [a for a in all_annotations if a['image_id'] not in test_image_ids]
    split_test_anns = [a for a in all_annotations if a['image_id'] in test_image_ids]

    print(f"📊 Random split (seed={seed}): {len(split_train_images)} train, {len(split_test_images)} test")
    print(f"   Train annotations: {len(split_train_anns)}, Test annotations: {len(split_test_anns)}")
    print(f"   Test images: {[img['file_name'] for img in split_test_images]}")

    # Save split train annotations
    split_train_path = os.path.join(annotations_dir, "split_train_annotations.json")
    with open(split_train_path, 'w') as f:
        json.dump({
            'info': data.get('info', {}), 'licenses': data.get('licenses', []),
            'images': split_train_images, 'annotations': split_train_anns,
            'categories': categories
        }, f, indent=2)

    # Save split test annotations (full images)
    split_test_path = os.path.join(annotations_dir, "split_test_annotations.json")
    with open(split_test_path, 'w') as f:
        json.dump({
            'info': data.get('info', {}), 'licenses': data.get('licenses', []),
            'images': split_test_images, 'annotations': split_test_anns,
            'categories': categories
        }, f, indent=2)

    # Crop test images into quadrants
    cropped_dir = os.path.join(dataset_root, "images_split_test_cropped")
    os.makedirs(cropped_dir, exist_ok=True)

    for img_info in split_test_images:
        img_path = os.path.join(images_dir, img_info['file_name'])
        img = Image.open(img_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        mid_x, mid_y = w // 2, h // 2
        stem = os.path.splitext(img_info['file_name'])[0]
        ext = os.path.splitext(img_info['file_name'])[1]

        quadrants = {'tl': (0, 0, mid_x, mid_y), 'tr': (mid_x, 0, w, mid_y),
                     'bl': (0, mid_y, mid_x, h), 'br': (mid_x, mid_y, w, h)}
        for qname, (x1, y1, x2, y2) in quadrants.items():
            crop = img_array[y1:y2, x1:x2] if len(img_array.shape) == 2 else img_array[y1:y2, x1:x2, :]
            Image.fromarray(crop).save(os.path.join(cropped_dir, f"{stem}_{qname}{ext}"))

    print(f"   Cropped {len(split_test_images)} test images → {len(split_test_images)*4} quadrants")

    # Create cropped test annotations
    from utils.create_cropped_annotations import create_cropped_annotations
    split_test_cropped_path = os.path.join(annotations_dir, "split_test_cropped_annotations.json")
    create_cropped_annotations(split_test_path, split_test_cropped_path, cropped_dir)

    # Register datasets
    register_coco_instances("injury_split_train", {}, split_train_path, images_dir)
    register_coco_instances("injury_split_val_cropped", {}, split_test_cropped_path, cropped_dir)

    print(f"   ✅ Registered injury_split_train ({len(split_train_images)} images)")
    print(f"   ✅ Registered injury_split_val_cropped ({len(split_test_images)*4} crops)")

    return "injury_split_train", "injury_split_val_cropped", split_test_images


def visualize_test_results(dataset_root, output_dir, config_file, best_model_path, split_info):
    """Run inference on test images with best model and create composed visualizations.

    Args:
        dataset_root: Path to dataset root
        output_dir: Training output directory
        config_file: Config yaml path
        best_model_path: Path to best model checkpoint
        split_info: dict with 'num_test', 'seed', 'test_images' (list of image info dicts)
    """
    import cv2
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    num_test = split_info['num_test']
    seed = split_info['seed']
    test_images = split_info['test_images']

    # Create identifiable output folder
    viz_dir = os.path.join(output_dir, f"viz_numtest{num_test}_seed{seed}")
    os.makedirs(viz_dir, exist_ok=True)

    # Save run info
    run_info = {
        'num_test': num_test,
        'seed': seed,
        'config': config_file,
        'model': best_model_path,
        'test_images': [img['file_name'] for img in test_images],
    }
    with open(os.path.join(viz_dir, "run_info.json"), 'w') as f:
        json.dump(run_info, f, indent=2)

    # Setup predictor
    print(f"\n🎨 Generating visualizations in {viz_dir}")
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = best_model_path
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.3
    cfg.freeze()

    predictor = DefaultPredictor(cfg)

    images_dir = os.path.join(dataset_root, "images")
    cropped_dir = os.path.join(dataset_root, "images_split_test_cropped")

    for img_info in test_images:
        fname = img_info['file_name']
        stem = os.path.splitext(fname)[0]
        ext = os.path.splitext(fname)[1]

        print(f"   Processing {fname}...")

        # Load original full image for dimensions
        orig_path = os.path.join(images_dir, fname)
        orig_img = np.array(Image.open(orig_path))
        h, w = orig_img.shape[:2]
        mid_x, mid_y = w // 2, h // 2

        quadrants = {
            'tl': (0, 0, mid_x, mid_y),
            'tr': (mid_x, 0, w, mid_y),
            'bl': (0, mid_y, mid_x, h),
            'br': (mid_x, mid_y, w, h),
        }

        # Compose overlay image at full resolution
        composed = np.zeros((h, w, 3), dtype=np.uint8)
        total_instances = 0

        for qname, (x1, y1, x2, y2) in quadrants.items():
            crop_path = os.path.join(cropped_dir, f"{stem}_{qname}{ext}")
            if not os.path.exists(crop_path):
                print(f"      ⚠️  Missing crop: {stem}_{qname}{ext}")
                continue

            # Load crop as BGR for predictor
            crop_bgr = cv2.imread(crop_path)
            if crop_bgr is None:
                # Try PIL for tiff
                crop_pil = np.array(Image.open(crop_path))
                if len(crop_pil.shape) == 2:
                    crop_bgr = cv2.cvtColor(crop_pil, cv2.COLOR_GRAY2BGR)
                else:
                    crop_bgr = cv2.cvtColor(crop_pil, cv2.COLOR_RGB2BGR)

            # Run inference
            with torch.no_grad():
                outputs = predictor(crop_bgr)

            instances = outputs["instances"].to("cpu")
            n_inst = len(instances)
            total_instances += n_inst

            # Create overlay on this crop
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            overlay = crop_rgb.copy()

            if n_inst > 0:
                masks = instances.pred_masks.numpy()
                # Generate distinct colors
                rng = np.random.RandomState(42)
                colors = rng.randint(60, 255, size=(n_inst, 3))

                for i in range(n_inst):
                    mask = masks[i].astype(bool)
                    color = colors[i]
                    # Semi-transparent fill
                    overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
                    # Draw contour
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

            # Place into composed image
            composed[y1:y2, x1:x2] = overlay

        # Draw quadrant boundary lines
        cv2.line(composed, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)
        cv2.line(composed, (0, mid_y), (w, mid_y), (255, 255, 255), 2)

        # Save composed image
        out_path = os.path.join(viz_dir, f"{stem}_overlay.png")
        Image.fromarray(composed).save(out_path)
        print(f"      ✅ Saved: {stem}_overlay.png ({total_instances} instances)")

    # Cleanup predictor
    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n✅ Visualizations saved to {viz_dir}")
    return viz_dir


def find_latest_checkpoint(output_dir: str) -> str:
    """Find the latest checkpoint in an output directory."""
    if not os.path.exists(output_dir):
        return ""

    # Look for model_final.pth first (completed training)
    final_model = os.path.join(output_dir, "model_final.pth")
    if os.path.exists(final_model):
        return final_model

    # Look for latest checkpoint
    checkpoint_pattern = os.path.join(output_dir, "model_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        return ""

    # Sort by modification time and return the latest
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def find_best_checkpoint(output_dir: str) -> str:
    """Find the best checkpoint based on evaluation metrics."""
    if not os.path.exists(output_dir):
        return ""

    best_patterns = [
        "model_best.pth",
        "best_model.pth",
        "model_best_*.pth"
    ]

    for pattern in best_patterns:
        best_files = glob.glob(os.path.join(output_dir, pattern))
        if best_files:
            best_files.sort(key=os.path.getmtime, reverse=True)
            return best_files[0]

    return find_latest_checkpoint(output_dir)


def count_dataset_stats(dataset_root):
    """Count images and annotations for the dataset."""
    annotations_dir = os.path.join(dataset_root, "annotations")
    train_ann = os.path.join(annotations_dir, "train_annotations.json")
    test_ann = os.path.join(annotations_dir, "test_annotations.json")

    stats = {
        'train_images': 0,
        'train_annotations': 0,
        'test_images': 0,
        'test_annotations': 0
    }

    if os.path.exists(train_ann):
        with open(train_ann, 'r') as f:
            data = json.load(f)
        stats['train_images'] = len(data['images'])
        stats['train_annotations'] = len(data['annotations'])

    if os.path.exists(test_ann):
        with open(test_ann, 'r') as f:
            data = json.load(f)
        stats['test_images'] = len(data['images'])
        stats['test_annotations'] = len(data['annotations'])

    return stats


def verify_dataset(dataset_root: str) -> bool:
    """Verify that dataset structure exists."""
    print(f"🔍 Verifying injury dataset...")

    issues = []

    if not os.path.exists(dataset_root):
        issues.append(f"Dataset root not found: {dataset_root}")

    images_dir = os.path.join(dataset_root, "images")
    if not os.path.exists(images_dir):
        issues.append(f"Missing images directory: {images_dir}")

    annotations_dir = os.path.join(dataset_root, "annotations")
    if not os.path.exists(annotations_dir):
        issues.append(f"Missing annotations directory: {annotations_dir}")
    else:
        train_ann = os.path.join(annotations_dir, "train_annotations.json")
        if not os.path.exists(train_ann):
            issues.append(f"Missing train annotations: train_annotations.json")

    if issues:
        print("❌ Dataset verification failed:")
        for issue in issues:
            print(f"   • {issue}")
        print(f"\n💡 Expected structure:")
        print(f"   {dataset_root}/")
        print(f"   ├── images/")
        print(f"   └── annotations/")
        print(f"       ├── train_annotations.json  (required)")
        print(f"       └── test_annotations.json   (optional)")
        return False

    print(f"✅ Dataset verified!")
    return True


def run_training(args, dataset_root, split_datasets=None):
    """Execute training on injury dataset."""
    stats = count_dataset_stats(dataset_root)

    config_file = args.config
    output_dir = args.output

    print("🚀 INJURY SEGMENTATION TRAINING")
    print("=" * 60)
    print(f"   Dataset: {dataset_root}")
    print(f"   Training: {stats['train_images']} images, {stats['train_annotations']} annotations")
    print(f"   Test: {stats['test_images']} images, {stats['test_annotations']} annotations")
    print(f"   Config: {config_file}")
    print(f"   Output: {output_dir}")
    print("=" * 60)

    # Prepare training arguments
    train_opts = []

    # Override output directory if specified
    if args.output != "./output_injury":
        train_opts.extend(["OUTPUT_DIR", args.output])

    # Override weights if specified
    if args.weights:
        train_opts.extend(["MODEL.WEIGHTS", args.weights])

    # Override max iterations if specified
    if args.max_iter:
        train_opts.extend(["SOLVER.MAX_ITER", str(args.max_iter)])

    # Override datasets if using split
    if split_datasets:
        train_name, test_name = split_datasets
        train_opts.extend(["DATASETS.TRAIN", f"('{train_name}',)",
                           "DATASETS.TEST", f"('{test_name}',)"])

    train_args = argparse.Namespace(
        config_file=config_file,
        num_gpus=args.num_gpus,
        resume=args.resume,
        eval_only=args.eval_only,
        opts=train_opts
    )

    # Start training
    if args.eval_only:
        print("🔄 Running evaluation...")
    else:
        print("🔄 Starting training...")

    try:
        train_main(train_args)

        if args.eval_only:
            print("✅ Evaluation completed!")
        else:
            print("✅ Training completed successfully!")

            # Check if checkpoint was created
            checkpoint = find_best_checkpoint(output_dir)
            if checkpoint:
                print(f"📄 Best model: {checkpoint}")
            else:
                checkpoint = find_latest_checkpoint(output_dir)
                if checkpoint:
                    print(f"📄 Latest checkpoint: {checkpoint}")

        return True

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for injury training."""
    parser = argparse.ArgumentParser(description="Injury segmentation training")

    # Dataset
    parser.add_argument("--dataset", default="injury_dataset",
                       help="Path to injury dataset directory")

    # Training control
    parser.add_argument("--config", default="injury_config.yaml",
                       help="Config file for training")
    parser.add_argument("--output", default="./output_injury",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true",
                       help="Perform evaluation only")

    # Model
    parser.add_argument("--weights", default=None,
                       help="Custom weights file (overrides config)")
    parser.add_argument("--max-iter", type=int, default=None,
                       help="Maximum iterations (overrides config)")

    # Split
    parser.add_argument("--num-test", type=int, default=0,
                       help="Number of images to hold out for test (will be cropped into 4 quadrants each)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for train/test split (default: 42)")

    args = parser.parse_args()

    # Print header
    print("🩹 Mask2Former Injury Segmentation Training")
    print("=" * 60)

    # Handle split or normal registration
    split_datasets = None
    split_test_images = None
    if args.num_test > 0:
        print(f"\n🔀 Splitting dataset: {args.num_test} images for test (× 4 quadrants = {args.num_test * 4} crops)")
        result = split_and_register(args.dataset, args.num_test, args.seed)
        if result is None:
            return 1
        train_name, test_name, split_test_images = result
        split_datasets = (train_name, test_name)
    else:
        # Register datasets normally
        print(f"\n🔄 Registering injury datasets...")
        registered = register_injury_datasets(args.dataset)
        if not registered:
            print("❌ No datasets registered. Please check your dataset structure.")
            return 1

    # Verify dataset
    if not args.eval_only:
        if not verify_dataset(args.dataset):
            return 1

    # Run training
    success = run_training(args, args.dataset, split_datasets=split_datasets)

    if success:
        print("\n" + "=" * 60)
        print("🎉 INJURY TRAINING COMPLETED!")
        print("=" * 60)

        if not args.eval_only:
            best_model = find_best_checkpoint(args.output)
            if best_model:
                print(f"   Best model: {best_model}")

            # Generate test set visualizations if we have a split
            if split_test_images and best_model:
                split_info = {
                    'num_test': args.num_test,
                    'seed': args.seed,
                    'test_images': split_test_images,
                }
                visualize_test_results(
                    args.dataset, args.output, args.config,
                    best_model, split_info
                )

            print(f"\n📋 Next steps:")
            print(f"   Evaluate: python train_injury.py --dataset {args.dataset} --eval-only")
            print(f"   Inference: Use the model in Fiji integration")

        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
