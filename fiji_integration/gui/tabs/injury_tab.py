"""
Injury segmentation tab for the Fiji integration GUI.

This tab runs Mask2Former injury segmentation. It internally crops each
input image into 4 quadrants (top-left, top-right, bottom-left, bottom-right),
runs inference on each crop, then reassembles the detected instances back
to original image coordinates.
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, Any, Optional
import threading

from fiji_integration.gui.base_tab import TabInterface
from fiji_integration.gui.output_stream import GUIOutputStream
from fiji_integration.utils.constants import DEFAULT_INJURY_GUI_CONFIG, IMAGE_EXTENSIONS
from fiji_integration.utils.path_utils import ensure_mask2former_loaded


__all__ = ['InjuryTab']


class InjuryTab(TabInterface):
    """Tab for injury instance segmentation with automatic quadrant cropping."""

    def __init__(self, config_file=None):
        super().__init__()

        if config_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fiji_integration_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            config_file = os.path.join(fiji_integration_dir, '.injury_gui_config.json')
        self.config_file = config_file

        home_dir = os.path.expanduser('~')
        workflow_base = os.path.join(home_dir, 'fiji_workflow')

        self.default_params = DEFAULT_INJURY_GUI_CONFIG.copy()
        self.default_params['input_path'] = os.path.join(workflow_base, '1_max_projection', 'myotube_channel')
        self.default_params['output_dir'] = os.path.join(workflow_base, '5_injury_segmentation')

        self.params = self.load_config()

        # GUI widgets
        self.input_var = None
        self.output_var = None
        self.config_var = None
        self.weights_var = None
        self.mask2former_path_var = None
        self.confidence_var = None
        self.min_area_var = None
        self.max_area_var = None
        self.final_min_area_var = None
        self.cpu_var = None
        self.save_measurements_var = None
        self.confidence_label = None
        self.run_button = None
        self.stop_button = None
        self.restore_button = None

    def get_tab_name(self) -> str:
        return "Injury Segmentation"

    def build_ui(self, parent_frame: ttk.Frame, console_text: tk.Text) -> None:
        self.console_text = console_text

        self.input_var = tk.StringVar(value=self.params['input_path'])
        self.output_var = tk.StringVar(value=self.params['output_dir'])
        self.config_var = tk.StringVar(value=self.params['config'])
        self.weights_var = tk.StringVar(value=self.params['weights'])
        self.mask2former_path_var = tk.StringVar(value=self.params['mask2former_path'])
        self.confidence_var = tk.DoubleVar(value=self.params['confidence'])
        self.min_area_var = tk.IntVar(value=self.params['min_area'])
        self.max_area_var = tk.IntVar(value=self.params['max_area'])
        self.final_min_area_var = tk.IntVar(value=self.params['final_min_area'])
        self.cpu_var = tk.BooleanVar(value=self.params['cpu'])
        self.save_measurements_var = tk.BooleanVar(value=self.params['save_measurements'])

        row = 0

        # ===== Info banner =====
        info_frame = ttk.LabelFrame(parent_frame, text="About", padding="5")
        info_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(info_frame, text="Each image is automatically split into 4 quadrants, "
                  "segmented individually, then reassembled to original coordinates.",
                  wraplength=700).pack(anchor=tk.W)
        row += 1

        # ===== Paths Section =====
        ttk.Label(parent_frame, text="Input/Output Paths", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Label(parent_frame, text="Input (Image/Directory):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.input_var, width=50).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_input).grid(row=row, column=2)
        row += 1

        ttk.Label(parent_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.output_var, width=50).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_output).grid(row=row, column=2)
        row += 1

        ttk.Separator(parent_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ===== Model Configuration =====
        ttk.Label(parent_frame, text="Model Configuration", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Label(parent_frame, text="*Config File:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.config_var, width=50).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_config).grid(row=row, column=2)
        row += 1

        ttk.Label(parent_frame, text="*Model Weights:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.weights_var, width=50).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_weights).grid(row=row, column=2)
        row += 1

        ttk.Label(parent_frame, text="*Mask2Former Path:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.mask2former_path_var, width=50).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_mask2former_path).grid(row=row, column=2)
        row += 1

        ttk.Separator(parent_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ===== Detection Parameters =====
        ttk.Label(parent_frame, text="Detection Parameters", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Label(parent_frame, text="Confidence Threshold (0-1):").grid(row=row, column=0, sticky=tk.W)
        confidence_scale = ttk.Scale(parent_frame, from_=0.0, to=1.0,
                                     variable=self.confidence_var, orient='horizontal', length=300)
        confidence_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        self.confidence_label = ttk.Label(parent_frame, text=f"{self.confidence_var.get():.2f}")
        self.confidence_label.grid(row=row, column=2)
        confidence_scale.configure(command=lambda v: self.confidence_label.configure(text=f"{float(v):.2f}"))
        row += 1

        ttk.Label(parent_frame, text="Minimum Area (pixels):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.min_area_var, width=20).grid(
            row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent_frame, text="Maximum Area (pixels):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.max_area_var, width=20).grid(
            row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent_frame, text="Final Min Area (pixels):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.final_min_area_var, width=20).grid(
            row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Separator(parent_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ===== Output Options =====
        ttk.Label(parent_frame, text="Output Options", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Checkbutton(parent_frame, text="Use CPU (slower but no GPU needed)",
                        variable=self.cpu_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Checkbutton(parent_frame, text="Save measurements CSV",
                        variable=self.save_measurements_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        parent_frame.columnconfigure(1, weight=1)

        # Buttons
        self.restore_button = ttk.Button(self.button_frame, text="Restore Defaults",
                                         command=self.restore_defaults)
        self.run_button = ttk.Button(self.button_frame, text="Run Injury Segmentation",
                                     command=self.on_run_threaded)
        self.stop_button = ttk.Button(self.button_frame, text="Stop",
                                      command=self.on_stop, state='disabled')

    def get_button_frame_widgets(self) -> list:
        return [
            (self.restore_button, tk.LEFT),
            (self.run_button, tk.LEFT),
            (self.stop_button, tk.LEFT),
        ]

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved = json.load(f)
                params = self.default_params.copy()
                params.update(saved)
                print(f"[LOADED] Loaded injury config from: {self.config_file}")
                return params
            except Exception as e:
                print(f"[WARNING] Could not load injury config: {e}")
                return self.default_params.copy()
        else:
            return self.default_params.copy()

    def save_config(self, config: Dict[str, Any] = None) -> None:
        if config is None:
            config = self.params
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[SAVED] Saved injury config to: {self.config_file}")
        except Exception as e:
            print(f"[WARNING] Could not save injury config: {e}")

    def validate_parameters(self) -> tuple[bool, Optional[str]]:
        self.update_params_from_gui()

        if not self.params['input_path']:
            return False, "Please select an input image or directory"
        if not self.params['output_dir']:
            return False, "Please select an output directory"

        try:
            if not (0 <= self.params['confidence'] <= 1):
                return False, "Confidence must be between 0 and 1"
            if self.params['min_area'] < 0:
                return False, "Minimum area must be non-negative"
            if self.params['max_area'] <= self.params['min_area']:
                return False, "Maximum area must be greater than minimum area"
            if self.params['final_min_area'] < 0:
                return False, "Final minimum area must be non-negative"
        except ValueError as e:
            return False, str(e)

        return True, None

    # --- GUI helpers ---

    def update_params_from_gui(self):
        self.params['input_path'] = self.input_var.get()
        self.params['output_dir'] = self.output_var.get()
        self.params['config'] = self.config_var.get()
        self.params['weights'] = self.weights_var.get()
        self.params['mask2former_path'] = self.mask2former_path_var.get()
        self.params['confidence'] = float(self.confidence_var.get())
        self.params['min_area'] = int(self.min_area_var.get())
        self.params['max_area'] = int(self.max_area_var.get())
        self.params['final_min_area'] = int(self.final_min_area_var.get())
        self.params['cpu'] = self.cpu_var.get()
        self.params['save_measurements'] = self.save_measurements_var.get()

    def update_gui_from_params(self):
        self.input_var.set(self.params['input_path'])
        self.output_var.set(self.params['output_dir'])
        self.config_var.set(self.params['config'])
        self.weights_var.set(self.params['weights'])
        self.mask2former_path_var.set(self.params['mask2former_path'])
        self.confidence_var.set(self.params['confidence'])
        self.min_area_var.set(self.params['min_area'])
        self.max_area_var.set(self.params['max_area'])
        self.final_min_area_var.set(self.params['final_min_area'])
        self.cpu_var.set(self.params['cpu'])
        self.save_measurements_var.set(self.params['save_measurements'])
        if self.confidence_label:
            self.confidence_label.configure(text=f"{self.params['confidence']:.2f}")

    def restore_defaults(self):
        self.params = self.default_params.copy()
        self.update_gui_from_params()
        print("[OK] Restored injury parameters to defaults")

    def browse_input(self):
        path = filedialog.askdirectory(title="Select Input Directory")
        if not path:
            path = filedialog.askopenfilename(
                title="Select Input Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
            )
        if path:
            self.input_var.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_var.set(path)

    def browse_config(self):
        path = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if path:
            self.config_var.set(path)

    def browse_weights(self):
        path = filedialog.askopenfilename(
            title="Select Model Weights",
            filetypes=[("Model files", "*.pth *.pkl"), ("All files", "*.*")]
        )
        if path:
            self.weights_var.set(path)

    def browse_mask2former_path(self):
        path = filedialog.askdirectory(title="Select Mask2Former Project Directory")
        if path:
            self.mask2former_path_var.set(path)

    # --- Execution ---

    def on_run_threaded(self):
        if self.is_running:
            messagebox.showwarning("Already Running", "Injury segmentation is already in progress.")
            return

        is_valid, error_msg = self.validate_parameters()
        if not is_valid:
            messagebox.showerror("Invalid Parameters", error_msg)
            return

        self.save_config()
        self.clear_console()
        self.write_to_console("=== Starting Injury Segmentation ===\n")
        self.write_to_console(f"Input: {self.params['input_path']}\n")
        self.write_to_console(f"Output: {self.params['output_dir']}\n")
        self.write_to_console(f"Mode: Auto-crop to 4 quadrants per image\n\n")

        self.is_running = True
        self.stop_requested = False
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')

        thread = threading.Thread(target=self.run_injury_segmentation)
        thread.daemon = True
        thread.start()

    def on_stop(self):
        if self.is_running:
            self.stop_requested = True
            self.write_to_console("\n[WARNING] Stop requested. Will halt after current image...\n")
            self.stop_button.config(state='disabled')

    def _auto_detect_injury_paths(self, m2f_path):
        """Auto-detect injury config and weights if not specified by user."""
        base = Path(m2f_path)

        if not self.params['config']:
            # Look for injury config
            candidates = [
                base / "output_injury" / "config.yaml",
                base / "injury_config.yaml",
            ]
            for c in candidates:
                if c.exists():
                    self.params['config'] = str(c)
                    print(f"   Auto-detected injury config: {c.name}")
                    break

        if not self.params['weights']:
            # Look for injury weights
            candidates = [
                base / "output_injury" / "model_best.pth",
                base / "output_injury" / "model_final.pth",
            ]
            for c in candidates:
                if c.exists():
                    self.params['weights'] = str(c)
                    print(f"   Auto-detected injury weights: {c.name}")
                    break

    def run_injury_segmentation(self):
        """Run injury segmentation with quadrant cropping."""
        old_stdout = sys.stdout

        try:
            sys.stdout = GUIOutputStream(self)

            print("[RUNNING] Loading Mask2Former and detectron2 modules...")
            m2f_path = ensure_mask2former_loaded(
                explicit_path=self.params.get('mask2former_path'))

            # Auto-detect injury-specific config/weights
            self._auto_detect_injury_paths(m2f_path)

            if not self.params['config']:
                print("[ERROR] No injury config found. Please specify a config file.")
                return
            if not self.params['weights']:
                print("[ERROR] No injury weights found. Please specify model weights.")
                return

            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config
            from detectron2.data.detection_utils import read_image
            from mask2former import add_maskformer2_config
            print("[OK] Modules loaded successfully\n")

            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            from myotube_segmentation import MyotubeFijiIntegration

            # Initialize the segmentation backend with injury config/weights
            integration = MyotubeFijiIntegration(
                config_file=self.params['config'],
                model_weights=self.params['weights'],
                skip_merged_masks=True,  # Always skip merged masks for injury (not applicable)
                mask2former_path=self.params.get('mask2former_path') or None
            )

            custom_config = {
                'confidence_threshold': self.params['confidence'],
                'min_area': self.params['min_area'],
                'max_area': self.params['max_area'],
                'final_min_area': self.params['final_min_area'],
                'force_cpu': self.params['cpu'],
                'save_measurements': self.params['save_measurements'],
            }

            # Initialize the predictor once (not per-quadrant)
            integration.initialize_predictor(force_cpu=self.params['cpu'])

            print(f"   Confidence threshold: {self.params['confidence']}")
            print(f"   Min area: {self.params['min_area']}, Max area: {self.params['max_area']}")

            # Collect images
            input_path = self.params['input_path']
            output_dir = self.params['output_dir']

            if os.path.isfile(input_path):
                image_files = [Path(input_path)]
            elif os.path.isdir(input_path):
                images_subdir = Path(input_path) / 'images'
                search_dir = images_subdir if images_subdir.is_dir() else Path(input_path)

                image_files_set = []
                for ext in IMAGE_EXTENSIONS:
                    for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                        image_files_set.extend(search_dir.rglob(pattern))

                unique = {}
                for f in image_files_set:
                    try:
                        resolved = str(f.resolve())
                        unique[resolved] = f
                    except (OSError, RuntimeError):
                        unique[str(f)] = f
                image_files = sorted(unique.values())
                print(f"[FOLDER] Found {len(image_files)} images in directory\n")
            else:
                print(f"[ERROR] Input path not found: {input_path}")
                return

            processed_count = 0
            for i, img_path in enumerate(image_files, 1):
                if self.stop_requested:
                    print(f"\nStopped by user after {processed_count}/{len(image_files)} images")
                    break

                print(f"\n{'='*60}")
                print(f"Processing {i}/{len(image_files)}: {img_path.name}")
                print(f"{'='*60}")

                base_name = img_path.stem
                image_output_dir = os.path.join(output_dir, base_name)

                try:
                    self._process_image_with_quadrants(
                        str(img_path), image_output_dir, integration, custom_config)
                    processed_count += 1
                except Exception as e:
                    print(f"[ERROR] Error processing {img_path.name}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    continue

            if not self.stop_requested:
                print(f"\n[OK] Injury segmentation complete! Processed {processed_count} images.")
            print(f"[LOADED] Results saved to: {output_dir}")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            # Always write error to log file so it survives a window crash
            log_path = os.path.join(output_dir, "injury_error.log")
            try:
                os.makedirs(output_dir, exist_ok=True)
                with open(log_path, 'w') as f:
                    f.write(f"Error: {type(e).__name__}: {str(e)}\n\n{error_msg}")
            except:
                pass
            try:
                print(f"\n{'='*80}")
                print(f"[ERROR] INJURY SEGMENTATION FAILED")
                print(f"{'='*80}")
                print(f"Error: {type(e).__name__}: {str(e)}")
                print(error_msg)
                print(f"Error log saved to: {log_path}")
            except:
                sys.stdout = old_stdout
                print(error_msg)
        finally:
            try:
                sys.stdout = old_stdout
            except:
                pass
            try:
                self.is_running = False
                self.root.after(0, lambda: self.run_button.config(state='normal'))
                self.root.after(0, lambda: self.stop_button.config(state='disabled'))
            except:
                pass

    @staticmethod
    def _preprocess_above130sqrt(image_path):
        """
        Apply Above130Sqrt preprocessing to match injury training data.
        Steps: read raw TIFF -> max project z-stack -> above-130 sqrt scaling -> 8-bit BGR.
        """
        import numpy as np
        import cv2

        try:
            import tifffile
            img = tifffile.imread(image_path)
        except ImportError:
            # Fallback: read with cv2 (won't handle z-stacks)
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            # cv2 reads as BGR; convert to grayscale if multi-channel
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print(f"      Raw image: shape={img.shape}, dtype={img.dtype}")

        # Max project if z-stack
        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            # Likely z-stack: (Z, H, W) where Z < H
            print(f"      Max projecting {img.shape[0]} slices...")
            img = img.max(axis=0)

        # If already 8-bit (e.g. PNG/JPEG), skip the sqrt scaling
        if img.dtype == np.uint8:
            print(f"      Image is already 8-bit, skipping Above130Sqrt")
            # Convert grayscale to BGR for predictor
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        # Auto-detect bit depth
        actual_max = int(img.max())
        if actual_max <= 8192:
            cam_max = 4095
            bit_depth = 12
        else:
            cam_max = 65535
            bit_depth = 16

        # Apply above-130 sqrt scaling
        BASE_THRESH_12BIT = 130
        BASE_CAM_MAX_12BIT = 4095
        threshold = round(BASE_THRESH_12BIT / BASE_CAM_MAX_12BIT * cam_max)

        img_f = img.astype(np.float32)
        out = np.zeros(img_f.shape, dtype=np.float32)
        mask = img_f > threshold
        out[mask] = np.sqrt((img_f[mask] - threshold) / (cam_max - threshold)) * 255
        out = np.clip(out, 0, 255).astype(np.uint8)

        n_nonzero = (out > 0).sum()
        print(f"      Above130Sqrt: {bit_depth}-bit, threshold={threshold}, "
              f"signal={n_nonzero / out.size * 100:.1f}%")

        # Convert grayscale to BGR for detectron2 predictor
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    def _process_image_with_quadrants(self, image_path, output_dir, integration, custom_config):
        """
        Crop image into 4 quadrants, run injury segmentation on each,
        then reassemble results back to original coordinates.
        Calls predictor directly to bypass myotube post-processing.
        """
        import cv2
        import numpy as np
        import torch

        os.makedirs(output_dir, exist_ok=True)

        # Preprocess: max project + Above130Sqrt to match training data
        print(f"   Preprocessing image (Above130Sqrt)...")
        preprocessed = self._preprocess_above130sqrt(image_path)
        if preprocessed is None:
            raise ValueError(f"Could not read image: {image_path}")

        orig_h, orig_w = preprocessed.shape[:2]
        print(f"   Preprocessed image size: {orig_w} x {orig_h}")

        crop_w, crop_h = orig_w // 2, orig_h // 2

        quadrants = [
            ('tl', 0,      0,      crop_w,            crop_h),
            ('tr', crop_w,  0,      orig_w - crop_w,   crop_h),
            ('bl', 0,      crop_h,  crop_w,            orig_h - crop_h),
            ('br', crop_w,  crop_h, orig_w - crop_w,   orig_h - crop_h),
        ]

        conf_thresh = custom_config.get('confidence_threshold', 0.05)
        min_area = custom_config.get('min_area', 30)
        max_area = custom_config.get('max_area', 50000)

        all_masks = []
        all_scores = []
        all_boxes = []

        for quad_name, x_off, y_off, q_w, q_h in quadrants:
            if self.stop_requested:
                break

            print(f"\n   --- Quadrant {quad_name.upper()} ({q_w}x{q_h}, offset [{x_off}, {y_off}]) ---")

            # Crop quadrant from preprocessed image
            crop_bgr = preprocessed[y_off:y_off + q_h, x_off:x_off + q_w].copy()

            # Run predictor directly on the crop (BGR format, as expected by detectron2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            predictions = integration._predictor(crop_bgr)
            instances = predictions["instances"]

            if len(instances) == 0:
                print(f"   Quadrant {quad_name}: 0 raw detections")
                continue

            scores = instances.scores.cpu().numpy()
            masks = instances.pred_masks.cpu().numpy()

            # Print score distribution for diagnostics
            print(f"   Quadrant {quad_name}: {len(scores)} raw detections")
            print(f"      Score distribution: min={scores.min():.4f}, max={scores.max():.4f}, "
                  f"median={np.median(scores):.4f}")
            print(f"      Scores > {conf_thresh}: {(scores >= conf_thresh).sum()}")

            # Simple filtering: confidence + area
            kept = 0
            for j in range(len(scores)):
                if scores[j] < conf_thresh:
                    continue

                mask = masks[j]
                area = mask.sum()
                if area < min_area or area > max_area:
                    continue

                # Place mask into full-image coordinates
                full_mask = np.zeros((orig_h, orig_w), dtype=bool)
                # Resize mask to quadrant size if needed (predictor may resize internally)
                if mask.shape[0] != q_h or mask.shape[1] != q_w:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask = cv2.resize(mask_uint8, (q_w, q_h),
                                      interpolation=cv2.INTER_NEAREST)
                    mask = mask > 128
                full_mask[y_off:y_off + q_h, x_off:x_off + q_w] = mask

                # Compute bounding box
                coords = np.where(full_mask)
                if len(coords[0]) == 0:
                    continue
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()

                all_masks.append(full_mask)
                all_scores.append(float(scores[j]))
                all_boxes.append(np.array([x_min, y_min, x_max, y_max]))
                kept += 1

            print(f"   Quadrant {quad_name}: {kept} injuries after filtering")

        if self.stop_requested:
            return

        print(f"\n   Total injuries across all quadrants: {len(all_masks)}")

        # Merge instances that span quadrant boundaries
        if len(all_masks) > 0:
            all_masks, all_scores, all_boxes = self._merge_boundary_instances(
                all_masks, all_scores, all_boxes, orig_h, orig_w, crop_w, crop_h)
            print(f"   After boundary merging: {len(all_masks)} injuries")

        # Build reassembled instances dict
        reassembled = {
            'masks': np.array(all_masks) if all_masks else np.array([]).reshape(0, orig_h, orig_w),
            'scores': np.array(all_scores) if all_scores else np.array([]),
            'boxes': np.array(all_boxes).reshape(-1, 4) if all_boxes else np.array([]).reshape(0, 4),
            'image_shape': (orig_h, orig_w),
        }

        self._save_reassembled_outputs(
            reassembled, preprocessed, image_path, output_dir, integration)

    def _merge_boundary_instances(self, masks, scores, boxes, orig_h, orig_w, crop_w, crop_h):
        """
        Merge injury instances that touch quadrant boundaries,
        as they likely represent the same injury split by the crop.
        """
        import numpy as np

        n = len(masks)
        if n <= 1:
            return masks, scores, boxes

        # Find masks that touch quadrant boundaries
        # Horizontal boundary at y=crop_h, vertical boundary at x=crop_w
        boundary_margin = 5  # pixels

        def touches_boundary(mask):
            """Check which boundaries a mask touches."""
            touches = set()
            # Check horizontal boundary (y = crop_h)
            region_h = mask[max(0, crop_h - boundary_margin):
                           min(orig_h, crop_h + boundary_margin), :]
            if region_h.any():
                touches.add('h')
            # Check vertical boundary (x = crop_w)
            region_v = mask[:, max(0, crop_w - boundary_margin):
                           min(orig_w, crop_w + boundary_margin)]
            if region_v.any():
                touches.add('v')
            return touches

        # Build groups of masks to merge
        boundary_info = [touches_boundary(m) for m in masks]

        # Use union-find to group overlapping boundary masks
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Check pairs that both touch the same boundary for overlap
        for i in range(n):
            if not boundary_info[i]:
                continue
            for j in range(i + 1, n):
                if not boundary_info[j]:
                    continue
                # Only merge if they share a boundary direction
                shared = boundary_info[i] & boundary_info[j]
                if not shared:
                    continue
                # Check if they overlap or are adjacent near the boundary
                intersection = np.logical_and(masks[i], masks[j])
                if intersection.any():
                    union(i, j)
                    continue
                # Check adjacency: dilate one mask slightly and check overlap
                from scipy.ndimage import binary_dilation
                dilated = binary_dilation(masks[i], iterations=3)
                if np.logical_and(dilated, masks[j]).any():
                    union(i, j)

        # Group by root
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        # Merge groups
        merged_masks = []
        merged_scores = []
        merged_boxes = []
        merges = 0

        for root, members in groups.items():
            if len(members) == 1:
                idx = members[0]
                merged_masks.append(masks[idx])
                merged_scores.append(scores[idx])
                merged_boxes.append(boxes[idx])
            else:
                # Merge all masks in this group
                combined = np.logical_or.reduce([masks[idx] for idx in members])
                best_score = max(scores[idx] for idx in members)

                coords = np.where(combined)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    box = np.array([x_min, y_min, x_max, y_max])
                else:
                    box = boxes[members[0]]

                merged_masks.append(combined)
                merged_scores.append(best_score)
                merged_boxes.append(box)
                merges += 1

        if merges > 0:
            print(f"   Merged {merges} groups at quadrant boundaries: {n} -> {len(merged_masks)}")

        return merged_masks, merged_scores, merged_boxes

    def _save_reassembled_outputs(self, instances, preprocessed_image, image_path, output_dir, integration):
        """Save the reassembled full-image results."""
        import cv2
        import numpy as np

        base_name = Path(image_path).stem
        num_instances = len(instances['masks'])

        # Save preprocessed image (Above130Sqrt, no overlay)
        processed_path = os.path.join(output_dir, f"{base_name}_processed.tif")
        cv2.imwrite(processed_path, preprocessed_image)
        print(f"   Saved preprocessed image: {os.path.basename(processed_path)}")

        # Save individual mask images
        masks_dir = os.path.join(output_dir, f"{base_name}_masks")
        os.makedirs(masks_dir, exist_ok=True)

        for i, mask in enumerate(instances['masks']):
            mask_path = os.path.join(masks_dir, f"Injury_{i+1}_mask.png")
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_uint8)

        print(f"   Saved {num_instances} injury masks to: {masks_dir}")

        # Save overlay visualization (drawn on preprocessed image)
        overlay_path = os.path.join(output_dir, f"{base_name}_injury_overlay.tif")
        self._save_injury_overlay(instances, preprocessed_image, overlay_path)

        # Save measurements CSV if requested
        if self.params.get('save_measurements', False) and num_instances > 0:
            measurements_path = os.path.join(output_dir, f"{base_name}_measurements.csv")
            self._save_injury_measurements(instances, measurements_path)

        # Save summary info
        info_path = os.path.join(output_dir, f"{base_name}_info.json")
        info = {
            'input_image': os.path.basename(image_path),
            'num_injuries': num_instances,
            'image_shape': list(preprocessed_image.shape[:2]),
            'config_file': self.params.get('config', ''),
            'model_weights': self.params.get('weights', ''),
            'method': 'quadrant_crop_reassemble',
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"   Output summary: {num_instances} injuries detected")

    def _save_injury_overlay(self, instances, original_image, output_path):
        """Save a colored overlay of injury detections on the original image."""
        import cv2
        import numpy as np

        overlay = original_image.copy()
        num = len(instances['masks'])

        if num == 0:
            cv2.imwrite(output_path, overlay)
            return

        # Generate distinct colors
        np.random.seed(42)
        colors = []
        for i in range(num):
            hue = int(180 * i / max(num, 1))
            hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            colors.append(tuple(int(c) for c in bgr))

        # Draw each mask
        alpha = 0.4
        for i, mask in enumerate(instances['masks']):
            color = colors[i % len(colors)]
            mask_bool = mask.astype(bool)

            # Colored overlay
            colored = np.zeros_like(overlay)
            colored[mask_bool] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)

            # Draw contour
            mask_uint8 = mask_bool.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        # Resize for memory if very large
        max_overlay_size = 3000
        h, w = overlay.shape[:2]
        if max(h, w) > max_overlay_size:
            scale = max_overlay_size / max(h, w)
            overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)

        cv2.imwrite(output_path, overlay)
        print(f"   Saved injury overlay: {os.path.basename(output_path)}")

    def _save_injury_measurements(self, instances, output_path):
        """Save basic injury measurements to CSV."""
        import numpy as np

        lines = ["Instance,Area,BBox_X,BBox_Y,BBox_Width,BBox_Height"]
        for i, (mask, box) in enumerate(zip(instances['masks'], instances['boxes'])):
            area = int(mask.sum())
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            lines.append(f"Injury_{i+1},{area},{box[0]},{box[1]},{bw},{bh}")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"   Saved measurements: {os.path.basename(output_path)}")
