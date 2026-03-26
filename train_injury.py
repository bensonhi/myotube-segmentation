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

# Import dataset registration
from register_injury_datasets import register_injury_datasets

# Import Mask2Former training
from train_net import main as train_main


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


def run_training(args, dataset_root):
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

    args = parser.parse_args()

    # Print header
    print("🩹 Mask2Former Injury Segmentation Training")
    print("=" * 60)

    # Register datasets
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
    success = run_training(args, args.dataset)

    if success:
        print("\n" + "=" * 60)
        print("🎉 INJURY TRAINING COMPLETED!")
        print("=" * 60)

        if not args.eval_only:
            best_model = find_best_checkpoint(args.output)
            if best_model:
                print(f"   Best model: {best_model}")

            print(f"\n📋 Next steps:")
            print(f"   Evaluate: python train_injury.py --dataset {args.dataset} --eval-only")
            print(f"   Inference: Use the model in Fiji integration")

        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
