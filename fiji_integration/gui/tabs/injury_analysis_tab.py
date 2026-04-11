"""
Injury-Myotube Analysis tab for the Fiji integration GUI.

This tab pairs pre-existing myotube and injury segmentation outputs,
computes per-injury area ratios and raw pixel intensities, and produces
CSVs, overlays, and summaries.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import threading
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import cv2
import pandas as pd

from fiji_integration.gui.base_tab import TabInterface
from fiji_integration.utils.constants import DEFAULT_INJURY_ANALYSIS_GUI_CONFIG, IMAGE_EXTENSIONS


__all__ = ['InjuryAnalysisTab']


class InjuryMyotubeAnalyzer:
    """Analyzes spatial relationships and intensity between injuries and myotubes."""

    def __init__(self, myotube_dir: str, injury_dir: str, original_images_dir: str,
                 output_dir: str, min_overlap_ratio: float = 0.1,
                 full_image_mode: bool = True, progress_callback=None):
        self.myotube_dir = Path(myotube_dir)
        self.injury_dir = Path(injury_dir)
        self.original_images_dir = Path(original_images_dir)
        self.output_dir = Path(output_dir)
        self.min_overlap_ratio = min_overlap_ratio
        self.full_image_mode = full_image_mode
        self.progress_callback = progress_callback

        self.all_injury_results: List[Dict] = []
        self.all_myotube_results: List[Dict] = []

    def log(self, message: str):
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # Mask I/O
    # ------------------------------------------------------------------

    def load_masks(self, folder: Path, prefix: str) -> Dict[int, np.ndarray]:
        """Load individual mask PNGs from a *_masks/ subdirectory."""
        masks_dir = folder / f"{folder.name}_masks"
        result: Dict[int, np.ndarray] = {}

        if not masks_dir.exists():
            self.log(f"  Warning: Masks directory not found: {masks_dir}")
            return result

        pattern = f"{prefix}_*_mask.png"
        mask_files = sorted(masks_dir.glob(pattern))

        for mask_file in mask_files:
            try:
                mask_id = int(mask_file.stem.split('_')[1])
            except (IndexError, ValueError):
                continue
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            if mask is not None:
                result[mask_id] = (mask > 0).astype(np.uint8)

        return result

    # ------------------------------------------------------------------
    # Original image loading (raw TIFF, 12/16-bit)
    # ------------------------------------------------------------------

    def find_original_image(self, sample_name: str) -> Optional[Path]:
        """Find the original raw image for a sample by stem name."""
        for ext in IMAGE_EXTENSIONS:
            candidate = self.original_images_dir / f"{sample_name}{ext}"
            if candidate.exists():
                return candidate

        # Recursive search as fallback
        for ext in IMAGE_EXTENSIONS:
            for p in self.original_images_dir.rglob(f"{sample_name}{ext}"):
                return p

        return None

    def load_raw_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load a raw TIFF preserving original bit-depth, max-project if z-stack."""
        try:
            import tifffile
            img = tifffile.imread(str(image_path))
        except ImportError:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 3 and img.shape[2] in (3, 4):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img is None:
            return None

        # Max-project z-stacks: shape (Z, H, W) where Z < H
        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            img = img.max(axis=0)

        # Collapse color channels if present
        if img.ndim == 3 and img.shape[2] in (3, 4):
            img = np.mean(img[:, :, :3], axis=2)

        return img.astype(np.float64)

    # ------------------------------------------------------------------
    # Sample matching
    # ------------------------------------------------------------------

    def find_matching_myotube_folder(self, injury_folder_name: str) -> Optional[Path]:
        """Find the myotube output folder that matches an injury sample."""
        # Direct name match
        candidate = self.myotube_dir / injury_folder_name
        if candidate.exists() and (candidate / f"{injury_folder_name}_info.json").exists():
            return candidate

        # Search all info.json files for a matching input_image stem
        for info_file in self.myotube_dir.rglob("*_info.json"):
            folder = info_file.parent
            if folder.name == injury_folder_name:
                return folder

        return None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def compute_intensity_stats(self, raw_image: np.ndarray,
                                mask: np.ndarray) -> Dict[str, float]:
        """Extract intensity statistics for pixels under a mask."""
        if raw_image.shape[:2] != mask.shape[:2]:
            mask_resized = cv2.resize(
                mask, (raw_image.shape[1], raw_image.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        values = raw_image[mask_resized > 0]

        if len(values) == 0:
            return {
                'mean_intensity': 0.0, 'median_intensity': 0.0,
                'max_intensity': 0.0, 'min_intensity': 0.0,
                'std_intensity': 0.0, 'cv_intensity': 0.0,
                'p25_intensity': 0.0, 'p75_intensity': 0.0,
            }

        mean_val = float(np.mean(values))
        std_val = float(np.std(values))

        return {
            'mean_intensity': mean_val,
            'median_intensity': float(np.median(values)),
            'max_intensity': float(np.max(values)),
            'min_intensity': float(np.min(values)),
            'std_intensity': std_val,
            'cv_intensity': std_val / mean_val if mean_val > 0 else 0.0,
            'p25_intensity': float(np.percentile(values, 25)),
            'p75_intensity': float(np.percentile(values, 75)),
        }

    def assign_injuries_to_myotubes(
        self, injury_masks: Dict[int, np.ndarray],
        myotube_masks: Dict[int, np.ndarray],
        myotube_areas: Dict[int, int],
    ) -> List[Dict]:
        """Assign each injury to its best-overlapping myotube."""
        # Build a labeled myotube image for fast lookup
        if not myotube_masks:
            ref_shape = next(iter(injury_masks.values())).shape
        else:
            ref_shape = next(iter(myotube_masks.values())).shape

        labeled_myotubes = np.zeros(ref_shape, dtype=np.int32)
        for mid, mmask in myotube_masks.items():
            if mmask.shape != ref_shape:
                mmask = cv2.resize(mmask, (ref_shape[1], ref_shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            labeled_myotubes[mmask > 0] = mid

        results = []
        for inj_id, inj_mask in injury_masks.items():
            if inj_mask.shape != ref_shape:
                inj_mask = cv2.resize(inj_mask, (ref_shape[1], ref_shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

            injury_area = int(np.sum(inj_mask > 0))
            if injury_area == 0:
                continue

            overlapping = labeled_myotubes[inj_mask > 0]
            overlapping = overlapping[overlapping > 0]

            if len(overlapping) == 0:
                results.append({
                    'injury_id': inj_id,
                    'injury_area_px': injury_area,
                    'assigned_myotube_id': None,
                    'myotube_area_px': 0,
                    'injury_area_ratio': 0.0,
                    'overlap_area_ratio': 0.0,
                    'overlap_pixels': 0,
                    'overlap_ratio': 0.0,
                    'assignment_status': 'unassigned',
                })
                continue

            unique_labels, counts = np.unique(overlapping, return_counts=True)
            best_idx = int(np.argmax(counts))
            best_myotube_id = int(unique_labels[best_idx])
            best_overlap_px = int(counts[best_idx])
            overlap_ratio = best_overlap_px / injury_area

            if overlap_ratio < self.min_overlap_ratio:
                myo_area = myotube_areas.get(best_myotube_id, 0)
                results.append({
                    'injury_id': inj_id,
                    'injury_area_px': injury_area,
                    'assigned_myotube_id': None,
                    'myotube_area_px': 0,
                    'injury_area_ratio': 0.0,
                    'overlap_area_ratio': best_overlap_px / myo_area if myo_area > 0 else 0.0,
                    'overlap_pixels': best_overlap_px,
                    'overlap_ratio': overlap_ratio,
                    'assignment_status': f'below_threshold ({overlap_ratio:.1%} < {self.min_overlap_ratio:.1%})',
                })
            else:
                myo_area = myotube_areas.get(best_myotube_id, 0)
                results.append({
                    'injury_id': inj_id,
                    'injury_area_px': injury_area,
                    'assigned_myotube_id': best_myotube_id,
                    'myotube_area_px': myo_area,
                    'injury_area_ratio': injury_area / myo_area if myo_area > 0 else 0.0,
                    'overlap_area_ratio': best_overlap_px / myo_area if myo_area > 0 else 0.0,
                    'overlap_pixels': best_overlap_px,
                    'overlap_ratio': overlap_ratio,
                    'assignment_status': 'assigned',
                })

        return results

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def create_overlay(self, sample_folder: Path, sample_name: str,
                       myotube_masks: Dict[int, np.ndarray],
                       injury_masks: Dict[int, np.ndarray],
                       assignment_results: List[Dict],
                       output_dir: Path):
        """Create an overlay showing myotube contours and injury fills."""
        # Try to load a base image for the overlay
        overlay_path = sample_folder / f"{sample_name}_processed_overlay.tif"
        if not overlay_path.exists():
            overlay_path = sample_folder / f"{sample_name}_injury_overlay.tif"
        if not overlay_path.exists():
            overlay_path = sample_folder / f"{sample_name}_processed.tif"

        if overlay_path.exists():
            base = cv2.imread(str(overlay_path))
        else:
            # Construct a blank image from mask dimensions
            ref = next(iter(myotube_masks.values())) if myotube_masks else next(iter(injury_masks.values()))
            base = np.zeros((*ref.shape[:2], 3), dtype=np.uint8)

        if base is None:
            return

        overlay = base.copy()
        h, w = overlay.shape[:2]

        # Draw myotube contours (cyan family)
        for mid, mmask in myotube_masks.items():
            if mmask.shape[:2] != (h, w):
                mmask = cv2.resize(mmask, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)  # cyan in BGR

            # Label myotube ID
            coords = np.where(mmask > 0)
            if len(coords[0]) > 0:
                cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
                cv2.putText(overlay, f"M{mid}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

        # Build assignment lookup
        assignment_map = {}
        for r in assignment_results:
            assignment_map[r['injury_id']] = r.get('assigned_myotube_id')

        # Draw injury masks (red=unassigned, magenta=assigned)
        for inj_id, inj_mask in injury_masks.items():
            if inj_mask.shape[:2] != (h, w):
                inj_mask = cv2.resize(inj_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            assigned = assignment_map.get(inj_id) is not None
            color = (255, 0, 255) if assigned else (0, 0, 255)  # magenta or red (BGR)

            # Semi-transparent fill
            fill_layer = overlay.copy()
            fill_layer[inj_mask > 0] = color
            cv2.addWeighted(fill_layer, 0.35, overlay, 0.65, 0, overlay)

            # Contour
            contours, _ = cv2.findContours(inj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Label injury ID
            coords = np.where(inj_mask > 0)
            if len(coords[0]) > 0:
                cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
                label = f"I{inj_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(overlay, (cx - 2, cy - th - 4), (cx + tw + 2, cy + 4), (0, 0, 0), -1)
                cv2.putText(overlay, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = output_dir / f"{sample_name}_injury_myotube_overlay.tif"
        cv2.imwrite(str(out_path), overlay)
        self.log(f"  Saved: {out_path.name}")

    # ------------------------------------------------------------------
    # Per-sample analysis
    # ------------------------------------------------------------------

    def analyze_sample(self, injury_folder: Path) -> bool:
        """Analyze a single sample: pair injury masks with myotube masks."""
        sample_name = injury_folder.name
        self.log(f"\nAnalyzing: {sample_name}")

        # Load injury info
        injury_info_file = injury_folder / f"{sample_name}_info.json"
        if not injury_info_file.exists():
            self.log(f"  Skipped: no injury _info.json in {injury_folder}")
            return False

        with open(injury_info_file, 'r') as f:
            injury_info = json.load(f)

        num_injuries = injury_info.get('num_injuries', 0)
        if num_injuries == 0:
            self.log(f"  Skipped: 0 injuries detected")
            return False

        # Load injury masks
        injury_masks = self.load_masks(injury_folder, "Injury")
        if not injury_masks:
            self.log(f"  Skipped: could not load injury masks")
            return False
        self.log(f"  Loaded {len(injury_masks)} injury masks")

        # Find matching myotube folder
        myotube_folder = self.find_matching_myotube_folder(sample_name)
        if myotube_folder is None:
            self.log(f"  Skipped: no matching myotube folder for '{sample_name}'")
            return False

        # Load myotube masks
        myotube_masks = self.load_masks(myotube_folder, "Myotube")
        if not myotube_masks:
            self.log(f"  Skipped: could not load myotube masks from {myotube_folder}")
            return False
        self.log(f"  Loaded {len(myotube_masks)} myotube masks")

        # Align dimensions: resize injury masks to myotube mask resolution
        ref_mask = next(iter(myotube_masks.values()))
        ref_shape = ref_mask.shape[:2]
        aligned_injury_masks: Dict[int, np.ndarray] = {}
        for iid, imask in injury_masks.items():
            if imask.shape[:2] != ref_shape:
                self.log(f"  Resizing injury mask {iid} from {imask.shape} to {ref_shape}")
                imask = cv2.resize(imask, (ref_shape[1], ref_shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            aligned_injury_masks[iid] = imask

        myotube_areas = {mid: int(np.sum(m > 0)) for mid, m in myotube_masks.items()}

        # Assign injuries to myotubes
        self.log(f"  Computing injury-myotube overlap...")
        assignment_results = self.assign_injuries_to_myotubes(
            aligned_injury_masks, myotube_masks, myotube_areas)

        # Load raw image for intensity extraction
        raw_image = None
        original_path = self.find_original_image(sample_name)
        if original_path is not None:
            self.log(f"  Loading raw image: {original_path.name}")
            raw_image = self.load_raw_image(original_path)
            if raw_image is not None:
                self.log(f"  Raw image: {raw_image.shape}, dtype range "
                         f"[{raw_image.min():.0f}, {raw_image.max():.0f}]")
        else:
            self.log(f"  Warning: no original image found for intensity extraction")

        # Enrich assignment results with intensity stats
        for result in assignment_results:
            iid = result['injury_id']
            inj_mask = aligned_injury_masks.get(iid)
            if raw_image is not None and inj_mask is not None:
                stats = self.compute_intensity_stats(raw_image, inj_mask)
            else:
                stats = self.compute_intensity_stats(np.zeros((1, 1)), np.zeros((1, 1), dtype=np.uint8))
            result.update(stats)

        # Build per-myotube summary
        myotube_summary = []
        for mid in sorted(myotube_masks.keys()):
            injuries_for_myo = [r for r in assignment_results
                                if r.get('assigned_myotube_id') == mid]
            total_inj_area = sum(r['injury_area_px'] for r in injuries_for_myo)
            myo_area = myotube_areas[mid]
            mean_inj_intensity = (
                float(np.mean([r['mean_intensity'] for r in injuries_for_myo]))
                if injuries_for_myo else 0.0)

            myotube_summary.append({
                'myotube_id': mid,
                'myotube_area_px': myo_area,
                'num_injuries': len(injuries_for_myo),
                'total_injury_area_px': total_inj_area,
                'injury_percentage': total_inj_area / myo_area * 100 if myo_area > 0 else 0.0,
                'mean_injury_intensity': mean_inj_intensity,
                'has_injuries': len(injuries_for_myo) > 0,
            })

        # Create output directory
        sample_output = self.output_dir / sample_name
        sample_output.mkdir(parents=True, exist_ok=True)

        # Save per-injury CSV
        inj_df = pd.DataFrame(assignment_results)
        inj_csv = sample_output / f"{sample_name}_injury_analysis.csv"
        inj_df.to_csv(inj_csv, index=False)
        self.log(f"  Saved: {inj_csv.name}")

        # Save per-myotube CSV
        myo_df = pd.DataFrame(myotube_summary)
        myo_csv = sample_output / f"{sample_name}_myotube_injury_summary.csv"
        myo_df.to_csv(myo_csv, index=False)
        self.log(f"  Saved: {myo_csv.name}")

        # Save overlay
        self.create_overlay(
            injury_folder, sample_name, myotube_masks,
            aligned_injury_masks, assignment_results, sample_output)

        # Save text summary
        self._write_sample_summary(
            sample_output / f"{sample_name}_injury_myotube_summary.txt",
            sample_name, inj_df, myo_df)

        # Accumulate for combined output
        for r in assignment_results:
            r['sample_name'] = sample_name
            self.all_injury_results.append(r)
        for r in myotube_summary:
            r['sample_name'] = sample_name
            self.all_myotube_results.append(r)

        assigned_count = sum(1 for r in assignment_results if r.get('assigned_myotube_id') is not None)
        self.log(f"  Result: {len(assignment_results)} injuries, "
                 f"{assigned_count} assigned to myotubes")
        return True

    # ------------------------------------------------------------------
    # Summary writers
    # ------------------------------------------------------------------

    def _write_sample_summary(self, path: Path, sample_name: str,
                              inj_df: pd.DataFrame, myo_df: pd.DataFrame):
        n_injuries = len(inj_df)
        n_myotubes = len(myo_df)
        assigned = inj_df[inj_df['assigned_myotube_id'].notna()]
        unassigned = inj_df[inj_df['assigned_myotube_id'].isna()]
        myo_with = myo_df[myo_df['num_injuries'] > 0]

        with open(path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"INJURY-MYOTUBE ANALYSIS SUMMARY\n")
            f.write(f"Sample: {sample_name}\n")
            f.write("=" * 80 + "\n\n")

            f.write("OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total injuries detected:         {n_injuries}\n")
            f.write(f"  Assigned to myotubes:          {len(assigned)}\n")
            f.write(f"  Unassigned:                    {len(unassigned)}\n")
            f.write(f"Total myotubes:                  {n_myotubes}\n")
            f.write(f"  With injuries:                 {len(myo_with)}\n")
            f.write(f"  Without injuries:              {n_myotubes - len(myo_with)}\n\n")

            if not assigned.empty:
                f.write("INJURY AREA RATIOS (assigned injuries)\n")
                f.write("-" * 80 + "\n")
                f.write(f"injury_area_ratio (total injury / myotube):\n")
                f.write(f"  Mean:                          {assigned['injury_area_ratio'].mean():.4f}\n")
                f.write(f"  Median:                        {assigned['injury_area_ratio'].median():.4f}\n")
                f.write(f"  Max:                           {assigned['injury_area_ratio'].max():.4f}\n")
                f.write(f"  Min:                           {assigned['injury_area_ratio'].min():.4f}\n")
                f.write(f"overlap_area_ratio (only inside-myotube pixels / myotube):\n")
                f.write(f"  Mean:                          {assigned['overlap_area_ratio'].mean():.4f}\n")
                f.write(f"  Median:                        {assigned['overlap_area_ratio'].median():.4f}\n")
                f.write(f"  Max:                           {assigned['overlap_area_ratio'].max():.4f}\n")
                f.write(f"  Min:                           {assigned['overlap_area_ratio'].min():.4f}\n\n")

            if 'mean_intensity' in inj_df.columns and n_injuries > 0:
                f.write("INTENSITY STATISTICS (all injuries)\n")
                f.write("-" * 80 + "\n")
                f.write(f"Mean intensity (avg across injuries):   {inj_df['mean_intensity'].mean():.1f}\n")
                f.write(f"Median intensity (avg):                 {inj_df['median_intensity'].mean():.1f}\n")
                f.write(f"Max intensity (max across injuries):    {inj_df['max_intensity'].max():.1f}\n")
                f.write(f"Coeff. of variation (avg):              {inj_df['cv_intensity'].mean():.3f}\n\n")

            if not myo_with.empty:
                f.write("PER-MYOTUBE INJURY BREAKDOWN\n")
                f.write("-" * 80 + "\n")
                f.write(f"Mean injury % per myotube:       {myo_with['injury_percentage'].mean():.2f}%\n")
                f.write(f"Max injury %:                    {myo_with['injury_percentage'].max():.2f}%\n\n")

            f.write("=" * 80 + "\n")

        self.log(f"  Saved: {path.name}")

    def _write_combined_summary(self, num_samples: int):
        """Write combined summary across all samples."""
        if not self.all_injury_results:
            return

        inj_df = pd.DataFrame(self.all_injury_results)
        myo_df = pd.DataFrame(self.all_myotube_results)

        assigned = inj_df[inj_df['assigned_myotube_id'].notna()]
        unassigned = inj_df[inj_df['assigned_myotube_id'].isna()]
        myo_with = myo_df[myo_df['num_injuries'] > 0]

        # Combined CSVs
        inj_df.to_csv(self.output_dir / "combined_injury_analysis.csv", index=False)
        myo_df.to_csv(self.output_dir / "combined_myotube_injury_summary.csv", index=False)
        self.log(f"Saved: combined_injury_analysis.csv")
        self.log(f"Saved: combined_myotube_injury_summary.csv")

        summary_path = self.output_dir / "combined_injury_myotube_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMBINED INJURY-MYOTUBE ANALYSIS SUMMARY (ALL SAMPLES)\n")
            f.write("=" * 80 + "\n\n")

            f.write("OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total samples analyzed:          {num_samples}\n")
            f.write(f"Total injuries:                  {len(inj_df)}\n")
            f.write(f"  Assigned to myotubes:          {len(assigned)}\n")
            f.write(f"  Unassigned:                    {len(unassigned)}\n")
            f.write(f"Total myotubes:                  {len(myo_df)}\n")
            f.write(f"  With injuries:                 {len(myo_with)}\n\n")

            if not assigned.empty:
                f.write("INJURY AREA RATIOS (all assigned injuries)\n")
                f.write("-" * 80 + "\n")
                f.write(f"injury_area_ratio (total injury / myotube):\n")
                f.write(f"  Mean:                          {assigned['injury_area_ratio'].mean():.4f}\n")
                f.write(f"  Median:                        {assigned['injury_area_ratio'].median():.4f}\n")
                f.write(f"  Max:                           {assigned['injury_area_ratio'].max():.4f}\n")
                f.write(f"  Min:                           {assigned['injury_area_ratio'].min():.4f}\n")
                f.write(f"overlap_area_ratio (only inside-myotube pixels / myotube):\n")
                f.write(f"  Mean:                          {assigned['overlap_area_ratio'].mean():.4f}\n")
                f.write(f"  Median:                        {assigned['overlap_area_ratio'].median():.4f}\n")
                f.write(f"  Max:                           {assigned['overlap_area_ratio'].max():.4f}\n")
                f.write(f"  Min:                           {assigned['overlap_area_ratio'].min():.4f}\n\n")

            if 'mean_intensity' in inj_df.columns and len(inj_df) > 0:
                f.write("INTENSITY STATISTICS (all injuries across samples)\n")
                f.write("-" * 80 + "\n")
                f.write(f"Mean intensity (avg):               {inj_df['mean_intensity'].mean():.1f}\n")
                f.write(f"Median intensity (avg):             {inj_df['median_intensity'].mean():.1f}\n")
                f.write(f"Max intensity (global max):         {inj_df['max_intensity'].max():.1f}\n")
                f.write(f"Coeff. of variation (avg):          {inj_df['cv_intensity'].mean():.3f}\n\n")

            # Per-sample breakdown table
            f.write("PER-SAMPLE BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            header = (f"{'Sample':<40} {'Injuries':>8} {'Assigned':>8} "
                      f"{'Myotubes':>8} {'MeanRatio':>10} {'MeanInt':>10}\n")
            f.write(header)
            f.write("-" * 80 + "\n")

            for sname in inj_df['sample_name'].unique():
                s_inj = inj_df[inj_df['sample_name'] == sname]
                s_assigned = s_inj[s_inj['assigned_myotube_id'].notna()]
                s_myo = myo_df[myo_df['sample_name'] == sname]
                mean_ratio = s_assigned['injury_area_ratio'].mean() if not s_assigned.empty else 0.0
                mean_int = s_inj['mean_intensity'].mean() if 'mean_intensity' in s_inj.columns else 0.0
                f.write(f"{sname:<40} {len(s_inj):>8} {len(s_assigned):>8} "
                        f"{len(s_myo):>8} {mean_ratio:>10.4f} {mean_int:>10.1f}\n")

            f.write("\n" + "=" * 80 + "\n")

        self.log(f"Saved: {summary_path.name}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def analyze_all_samples(self):
        """Iterate over injury output folders and analyze each sample."""
        injury_folders = []
        for info_file in sorted(self.injury_dir.rglob("*_info.json")):
            folder = info_file.parent
            if not folder.name.startswith('.') and folder != self.injury_dir:
                injury_folders.append(folder)

        # Deduplicate (same folder could match multiple info files)
        seen = set()
        unique_folders = []
        for f in injury_folders:
            if f not in seen:
                seen.add(f)
                unique_folders.append(f)
        injury_folders = unique_folders

        self.log(f"Found {len(injury_folders)} injury sample(s) to analyze")
        self.log(f"Myotube results in: {self.myotube_dir}")
        self.log(f"Original images in: {self.original_images_dir}")
        self.log(f"Min overlap ratio:  {self.min_overlap_ratio:.0%}")
        self.log("-" * 80)

        success = 0
        skipped = 0

        for i, folder in enumerate(injury_folders, 1):
            self.log(f"[{i}/{len(injury_folders)}]")
            if self.analyze_sample(folder):
                success += 1
            else:
                skipped += 1

        self.log("-" * 80)
        self.log(f"ANALYSIS COMPLETE: {success} succeeded, {skipped} skipped "
                 f"out of {len(injury_folders)} samples")

        if success > 0:
            self.log("-" * 80)
            self.log("Writing combined summary...")
            self._write_combined_summary(success)


# ======================================================================
# GUI Tab
# ======================================================================

class InjuryAnalysisTab(TabInterface):
    """Tab for injury-myotube relationship analysis."""

    def __init__(self, config_file=None):
        super().__init__()

        if config_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fiji_integration_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            config_file = os.path.join(fiji_integration_dir, '.injury_analysis_gui_config.json')
        self.config_file = config_file

        home_dir = os.path.expanduser('~')
        workflow_base = os.path.join(home_dir, 'fiji_workflow')

        self.default_params = DEFAULT_INJURY_ANALYSIS_GUI_CONFIG.copy()
        self.default_params['myotube_folder'] = os.path.join(workflow_base, '2_myotube_segmentation')
        self.default_params['injury_folder'] = os.path.join(workflow_base, '5_injury_segmentation')
        self.default_params['original_images_folder'] = os.path.join(workflow_base, '1_max_projection', 'myotube_channel')
        self.default_params['output_folder'] = os.path.join(workflow_base, '6_injury_myotube_analysis')

        self.params = self.load_config()

        # GUI variables (created in build_ui)
        self.myotube_folder_var = None
        self.injury_folder_var = None
        self.original_images_var = None
        self.output_folder_var = None
        self.min_overlap_var = None
        self.full_image_mode_var = None
        self.run_button = None
        self.stop_button = None
        self.restore_button = None

    def get_tab_name(self) -> str:
        return "Injury-Myotube Analysis"

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved = json.load(f)
                params = self.default_params.copy()
                params.update(saved)
                return params
            except Exception:
                return self.default_params.copy()
        return self.default_params.copy()

    def save_config(self, config=None):
        if config is None:
            config = self.params
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass

    def validate_parameters(self):
        myotube = self.params['myotube_folder'].strip()
        injury = self.params['injury_folder'].strip()
        original = self.params['original_images_folder'].strip()

        if not myotube:
            return False, "Please select a myotube results folder"
        if not injury:
            return False, "Please select an injury results folder"
        if not original:
            return False, "Please select the original images folder"
        if not os.path.exists(myotube):
            return False, f"Myotube folder does not exist: {myotube}"
        if not os.path.exists(injury):
            return False, f"Injury folder does not exist: {injury}"
        if not os.path.exists(original):
            return False, f"Original images folder does not exist: {original}"

        return True, None

    def build_ui(self, parent_frame, console_text):
        self.console_text = console_text

        self.myotube_folder_var = tk.StringVar(value=self.params['myotube_folder'])
        self.injury_folder_var = tk.StringVar(value=self.params['injury_folder'])
        self.original_images_var = tk.StringVar(value=self.params['original_images_folder'])
        self.output_folder_var = tk.StringVar(value=self.params['output_folder'])
        self.min_overlap_var = tk.StringVar(value=str(self.params['min_overlap_ratio']))
        self.full_image_mode_var = tk.BooleanVar(value=self.params['full_image_mode'])

        # --- Input/Output Section ---
        io_frame = ttk.LabelFrame(parent_frame, text="Input/Output", padding=10)
        io_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        ttk.Label(io_frame, text="Myotube Results Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(io_frame, textvariable=self.myotube_folder_var, width=50).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse",
                   command=lambda: self._browse_folder(self.myotube_folder_var, "Select Myotube Results Folder")
                   ).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(io_frame, text="Injury Results Folder:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(io_frame, textvariable=self.injury_folder_var, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse",
                   command=lambda: self._browse_folder(self.injury_folder_var, "Select Injury Results Folder")
                   ).grid(row=1, column=2, padx=5, pady=2)

        ttk.Label(io_frame, text="Original Images Folder:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(io_frame, textvariable=self.original_images_var, width=50).grid(
            row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse",
                   command=lambda: self._browse_folder(self.original_images_var, "Select Original Images Folder")
                   ).grid(row=2, column=2, padx=5, pady=2)

        ttk.Label(io_frame, text="Output Folder:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(io_frame, textvariable=self.output_folder_var, width=50).grid(
            row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse",
                   command=lambda: self._browse_folder(self.output_folder_var, "Select Output Folder")
                   ).grid(row=3, column=2, padx=5, pady=2)

        io_frame.columnconfigure(1, weight=1)

        # --- Parameters Section ---
        param_frame = ttk.LabelFrame(parent_frame, text="Analysis Parameters", padding=10)
        param_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        ttk.Label(param_frame, text="Min Overlap Ratio (0-1):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(param_frame, textvariable=self.min_overlap_var, width=10).grid(
            row=0, column=1, sticky=tk.W, pady=2)

        # --- Options Section ---
        options_frame = ttk.LabelFrame(parent_frame, text="Processing Options", padding=10)
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        ttk.Checkbutton(options_frame,
                        text="Full Image Mode (process complete images without quadrant cropping)",
                        variable=self.full_image_mode_var).grid(row=0, column=0, sticky=tk.W, pady=2)

        # --- Info ---
        info_frame = ttk.LabelFrame(parent_frame, text="About", padding=5)
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(info_frame,
                  text=("Pairs injury segmentation masks with myotube segmentation masks to compute:\n"
                        "  1) Injury percentage per myotube (area ratio)\n"
                        "  2) Raw pixel intensity within each lesion (mean, median, max, std, CV)"),
                  wraplength=700, justify=tk.LEFT).pack(anchor=tk.W)

        # Buttons
        self.run_button = ttk.Button(self.button_frame, text="Run Analysis", command=self._on_run_threaded)
        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self._on_stop, state='disabled')
        self.restore_button = ttk.Button(self.button_frame, text="Restore Defaults", command=self._restore_defaults)

    def get_button_frame_widgets(self):
        return [
            (self.restore_button, tk.LEFT),
            (self.run_button, tk.LEFT),
            (self.stop_button, tk.LEFT),
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _browse_folder(self, var: tk.StringVar, title: str):
        folder = filedialog.askdirectory(title=title)
        if folder:
            var.set(folder)

    def _update_params_from_gui(self):
        self.params['myotube_folder'] = self.myotube_folder_var.get()
        self.params['injury_folder'] = self.injury_folder_var.get()
        self.params['original_images_folder'] = self.original_images_var.get()
        self.params['output_folder'] = self.output_folder_var.get()
        self.params['min_overlap_ratio'] = float(self.min_overlap_var.get())
        self.params['full_image_mode'] = self.full_image_mode_var.get()

    def _restore_defaults(self):
        self.params = self.default_params.copy()
        self.myotube_folder_var.set(self.params['myotube_folder'])
        self.injury_folder_var.set(self.params['injury_folder'])
        self.original_images_var.set(self.params['original_images_folder'])
        self.output_folder_var.set(self.params['output_folder'])
        self.min_overlap_var.set(str(self.params['min_overlap_ratio']))
        self.full_image_mode_var.set(self.params['full_image_mode'])
        self._log("Parameters restored to defaults")

    def _log(self, message: str):
        self.write_to_console(message + "\n")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _on_run_threaded(self):
        if self.is_running:
            return

        try:
            self._update_params_from_gui()
        except ValueError as e:
            self._log(f"Error: Invalid parameter value: {e}")
            return

        valid, error_msg = self.validate_parameters()
        if not valid:
            self._log(f"Error: {error_msg}")
            return

        self.save_config()

        self.is_running = True
        self.stop_requested = False
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self):
        try:
            self._log("=" * 80)
            self._log("INJURY-MYOTUBE RELATIONSHIP ANALYSIS")
            self._log("=" * 80)
            self._log(f"Myotube folder:    {self.params['myotube_folder']}")
            self._log(f"Injury folder:     {self.params['injury_folder']}")
            self._log(f"Original images:   {self.params['original_images_folder']}")
            self._log(f"Output folder:     {self.params['output_folder']}")
            self._log(f"Min overlap ratio: {self.params['min_overlap_ratio']}")
            self._log(f"Full image mode:   {self.params['full_image_mode']}")
            self._log("=" * 80)

            analyzer = InjuryMyotubeAnalyzer(
                myotube_dir=self.params['myotube_folder'],
                injury_dir=self.params['injury_folder'],
                original_images_dir=self.params['original_images_folder'],
                output_dir=self.params['output_folder'],
                min_overlap_ratio=self.params['min_overlap_ratio'],
                full_image_mode=self.params['full_image_mode'],
                progress_callback=self._log,
            )

            analyzer.analyze_all_samples()

            self._log("\n" + "=" * 80)
            self._log("Analysis complete!")
            self._log(f"Results saved in: {self.params['output_folder']}")
            self._log("\nPer sample:")
            self._log("  - {sample}_injury_analysis.csv        (per-injury metrics)")
            self._log("  - {sample}_myotube_injury_summary.csv (per-myotube metrics)")
            self._log("  - {sample}_injury_myotube_overlay.tif (visualization)")
            self._log("  - {sample}_injury_myotube_summary.txt (text summary)")
            self._log("\nOverlay color coding:")
            self._log("  CYAN contours:  Myotube boundaries")
            self._log("  MAGENTA fill:   Injuries assigned to a myotube")
            self._log("  RED fill:       Unassigned injuries")
            self._log("=" * 80)

        except Exception as e:
            self._log(f"Error during analysis: {e}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            self.is_running = False
            self.run_button.config(state='normal')
            self.stop_button.config(state='disabled')

    def _on_stop(self):
        if self.is_running:
            self.stop_requested = True
            self._log("\nStop requested...")
