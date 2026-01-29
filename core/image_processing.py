"""
Lumina Studio - Image Processing Core
Image processing core module - Handles image loading, preprocessing, color quantization and matching
"""

import numpy as np
import cv2
from PIL import Image
from scipy.spatial import KDTree

from config import PrinterConfig


class LuminaImageProcessor:
    """
    Image processor class
    Handles LUT loading, image processing, and color matching
    """
    
    def __init__(self, lut_path, color_mode):
        """
        Initialize image processor
        
        Args:
            lut_path: LUT file path (.npy)
            color_mode: Color mode string (CMYW/RYBW)
        """
        self.color_mode = color_mode
        self.lut_rgb = None
        self.ref_stacks = None
        self.kdtree = None
        
        # Load and validate LUT
        self._load_lut(lut_path)
    
    def _load_lut(self, lut_path):
        """Load and validate LUT file"""
        try:
            lut_grid = np.load(lut_path)
            measured_colors = lut_grid.reshape(-1, 3)
        except Exception as e:
            raise ValueError(f"❌ LUT file corrupted: {e}")
        
        valid_rgb, valid_stacks = [], []
        base_blue = np.array([30, 100, 200])
        dropped = 0
        
        # Filter outliers
        for i in range(1024):
            digits = []
            temp = i
            for _ in range(5):
                digits.append(temp % 4)
                temp //= 4
            stack = digits[::-1]
            
            real_rgb = measured_colors[i]
            dist = np.linalg.norm(real_rgb - base_blue)
            
            # Filter out anomalies: close to blue but doesn't contain blue
            if dist < 60 and 3 not in stack:
                dropped += 1
                continue
            
            valid_rgb.append(real_rgb)
            valid_stacks.append(stack)
        
        self.lut_rgb = np.array(valid_rgb)
        self.ref_stacks = np.array(valid_stacks)
        self.kdtree = KDTree(self.lut_rgb)
        
        print(f"✅ LUT loaded (filtered {dropped} outliers)")
    
    def process_image(self, image_path, target_width_mm, modeling_mode,
                     quantize_colors, auto_bg, bg_tol,
                     blur_kernel=0, smooth_sigma=10):
        """
        Main image processing method
        
        Args:
            image_path: Image file path
            target_width_mm: Target width (millimeters)
            modeling_mode: Modeling mode ("high-fidelity", "pixel")
            quantize_colors: K-Means quantization color count
            auto_bg: Whether to auto-remove background
            bg_tol: Background tolerance
            blur_kernel: Median filter kernel size (0=disabled, recommended 0-5)
            smooth_sigma: Bilateral filter sigma value (recommended 5-20)
        
        Returns:
            dict: Dictionary containing processing results
                - matched_rgb: (H, W, 3) Matched RGB array
                - material_matrix: (H, W, Layers) Material index matrix
                - mask_solid: (H, W) Solid mask
                - dimensions: (width, height) Pixel dimensions
                - pixel_scale: mm/pixel ratio
                - mode_info: Mode information dictionary
                - debug_data: Debug data (high-fidelity mode only)
        """
        # Normalize modeling mode
        mode_str = str(modeling_mode).lower()
        use_high_fidelity = "high-fidelity" in mode_str or "高保真" in mode_str
        use_pixel = "pixel" in mode_str or "像素" in mode_str
        
        # Determine mode name
        if use_high_fidelity:
            mode_name = "High-Fidelity"
        elif use_pixel:
            mode_name = "Pixel Art"
        else:
            # Default to High-Fidelity if mode is unclear
            mode_name = "High-Fidelity"
            use_high_fidelity = True
        
        print(f"[IMAGE_PROCESSOR] Mode: {mode_name}")
        print(f"[IMAGE_PROCESSOR] Filter settings: blur_kernel={blur_kernel}, smooth_sigma={smooth_sigma}")
        
        # Load image
        img = Image.open(image_path).convert('RGBA')
        
        # DEBUG: Check original image properties
        print(f"[IMAGE_PROCESSOR] Original image: {image_path}")
        print(f"[IMAGE_PROCESSOR] Image mode: {Image.open(image_path).mode}")
        print(f"[IMAGE_PROCESSOR] Image size: {Image.open(image_path).size}")
        
        # Check if image has transparency
        original_img = Image.open(image_path)
        has_alpha = original_img.mode in ('RGBA', 'LA') or (original_img.mode == 'P' and 'transparency' in original_img.info)
        print(f"[IMAGE_PROCESSOR] Has alpha channel: {has_alpha}")
        
        if has_alpha:
            # Check alpha channel statistics
            if original_img.mode != 'RGBA':
                original_img = original_img.convert('RGBA')
            alpha_data = np.array(original_img)[:, :, 3]
            print(f"[IMAGE_PROCESSOR] Alpha stats: min={alpha_data.min()}, max={alpha_data.max()}, mean={alpha_data.mean():.1f}")
            print(f"[IMAGE_PROCESSOR] Transparent pixels (alpha<10): {np.sum(alpha_data < 10)}")
        
        # Calculate target resolution
        if use_high_fidelity:
            # High-precision mode: 10 pixels/mm
            PIXELS_PER_MM = 10
            target_w = int(target_width_mm * PIXELS_PER_MM)
            pixel_to_mm_scale = 1.0 / PIXELS_PER_MM  # 0.1 mm per pixel
            print(f"[IMAGE_PROCESSOR] High-res mode: {PIXELS_PER_MM} px/mm")
        else:
            # Pixel mode: Based on nozzle width
            target_w = int(target_width_mm / PrinterConfig.NOZZLE_WIDTH)
            pixel_to_mm_scale = PrinterConfig.NOZZLE_WIDTH
            print(f"[IMAGE_PROCESSOR] Pixel mode: {1.0/pixel_to_mm_scale:.2f} px/mm")
        
        target_h = int(target_w * img.height / img.width)
        print(f"[IMAGE_PROCESSOR] Target: {target_w}×{target_h}px ({target_w*pixel_to_mm_scale:.1f}×{target_h*pixel_to_mm_scale:.1f}mm)")
        
        # ========== CRITICAL FIX: Use NEAREST for both modes ==========
        # REASON: LANCZOS anti-aliasing creates light transition pixels at edges.
        # These light pixels map to stacks with WHITE bases (Layer 1),
        # causing the mesh to "float" above the build plate.
        # 
        # SOLUTION: Use NEAREST to preserve hard edges and ensure dark pixels
        # map to solid dark stacks from Layer 1 upwards.
        print(f"[IMAGE_PROCESSOR] Using NEAREST interpolation (no anti-aliasing)")
        img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
        
        img_arr = np.array(img)
        rgb_arr = img_arr[:, :, :3]
        alpha_arr = img_arr[:, :, 3]
        
        # CRITICAL FIX: Identify transparent pixels BEFORE color processing
        # This prevents transparent areas from being matched to LUT colors
        mask_transparent_initial = alpha_arr < 10
        print(f"[IMAGE_PROCESSOR] Found {np.sum(mask_transparent_initial)} transparent pixels (alpha<10)")
        
        # Color processing and matching
        debug_data = None
        if use_high_fidelity:
            matched_rgb, material_matrix, bg_reference, debug_data = self._process_high_fidelity_mode(
                rgb_arr, target_h, target_w, quantize_colors, blur_kernel, smooth_sigma
            )
        else:
            matched_rgb, material_matrix, bg_reference = self._process_pixel_mode(
                rgb_arr, target_h, target_w
            )
        
        # Background removal - combine alpha transparency with optional auto-bg
        mask_transparent = mask_transparent_initial.copy()
        if auto_bg:
            bg_color = bg_reference[0, 0]
            diff = np.sum(np.abs(bg_reference - bg_color), axis=-1)
            mask_transparent = np.logical_or(mask_transparent, diff < bg_tol)
        
        # Apply transparency mask to material matrix
        material_matrix[mask_transparent] = -1
        mask_solid = ~mask_transparent
        
        result = {
            'matched_rgb': matched_rgb,
            'material_matrix': material_matrix,
            'mask_solid': mask_solid,
            'dimensions': (target_w, target_h),
            'pixel_scale': pixel_to_mm_scale,
            'mode_info': {
                'name': mode_name,
                'use_high_fidelity': use_high_fidelity,
                'use_pixel': use_pixel
            }
        }
        
        # Add debug data (high-fidelity mode only)
        if debug_data is not None:
            result['debug_data'] = debug_data
        
        return result

    
    def _process_high_fidelity_mode(self, rgb_arr, target_h, target_w, quantize_colors,
                                    blur_kernel, smooth_sigma):
        """
        High-fidelity mode image processing
        Includes configurable filtering, K-Means quantization and color matching
        
        Args:
            rgb_arr: Input RGB array
            target_h: Target height
            target_w: Target width
            quantize_colors: K-Means color count
            blur_kernel: Median filter kernel size (0=disabled)
            smooth_sigma: Bilateral filter sigma value
        
        Returns:
            tuple: (matched_rgb, material_matrix, quantized_image, debug_data)
        """
        print(f"[IMAGE_PROCESSOR] Starting edge-preserving processing...")
        
        # Step 1: Bilateral filter (edge-preserving smoothing)
        if smooth_sigma > 0:
            print(f"[IMAGE_PROCESSOR] Applying bilateral filter (sigma={smooth_sigma})...")
            rgb_processed = cv2.bilateralFilter(
                rgb_arr.astype(np.uint8), 
                d=9,  # Larger neighborhood for better smoothing
                sigmaColor=smooth_sigma, 
                sigmaSpace=smooth_sigma
            )
        else:
            print(f"[IMAGE_PROCESSOR] Bilateral filter disabled (sigma=0)")
            rgb_processed = rgb_arr.astype(np.uint8)
        
        # Step 2: Optional median filter (remove salt-and-pepper noise)
        if blur_kernel > 0:
            # Ensure kernel size is odd
            kernel_size = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            print(f"[IMAGE_PROCESSOR] Applying median blur (kernel={kernel_size})...")
            rgb_processed = cv2.medianBlur(rgb_processed, kernel_size)
        else:
            print(f"[IMAGE_PROCESSOR] Median blur disabled (kernel=0)")
        
        # Step 3: Optional sharpening (enhance contours)
        # Use gentle sharpening kernel to enhance details
        sharpen_kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ])
        print(f"[IMAGE_PROCESSOR] Applying subtle sharpening...")
        rgb_sharpened = cv2.filter2D(rgb_processed, -1, sharpen_kernel)
        rgb_sharpened = np.clip(rgb_sharpened, 0, 255).astype(np.uint8)
        
        # Step 4: K-Means quantization
        print(f"[IMAGE_PROCESSOR] K-Means quantization to {quantize_colors} colors...")
        h, w = rgb_sharpened.shape[:2]
        pixels = rgb_sharpened.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_PP_CENTERS
        
        _, labels, centers = cv2.kmeans(
            pixels, quantize_colors, None, criteria, 10, flags
        )
        
        centers = centers.astype(np.uint8)
        quantized_pixels = centers[labels.flatten()]
        quantized_image = quantized_pixels.reshape(h, w, 3)
        
        print(f"[IMAGE_PROCESSOR] Quantization complete!")
        
        # Find unique colors
        unique_colors = np.unique(quantized_image.reshape(-1, 3), axis=0)
        print(f"[IMAGE_PROCESSOR] Found {len(unique_colors)} unique colors")
        
        # Match to LUT
        print(f"[IMAGE_PROCESSOR] Matching colors to LUT...")
        _, unique_indices = self.kdtree.query(unique_colors.astype(float))
        
        # Build color mapping
        color_to_stack = {}
        color_to_rgb = {}
        for i, color in enumerate(unique_colors):
            color_key = tuple(color)
            color_to_stack[color_key] = self.ref_stacks[unique_indices[i]]
            color_to_rgb[color_key] = self.lut_rgb[unique_indices[i]]
        
        # Map back to full image
        print(f"[IMAGE_PROCESSOR] Mapping to full image...")
        matched_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        material_matrix = np.zeros((target_h, target_w, PrinterConfig.COLOR_LAYERS), dtype=int)
        
        for y in range(target_h):
            for x in range(target_w):
                color_key = tuple(quantized_image[y, x])
                matched_rgb[y, x] = color_to_rgb[color_key]
                material_matrix[y, x] = color_to_stack[color_key]
        
        print(f"[IMAGE_PROCESSOR] Color matching complete!")
        
        # Prepare debug data
        debug_data = {
            'quantized_image': quantized_image.copy(),  # K-Means quantized image
            'num_colors': len(unique_colors),
            'bilateral_filtered': rgb_processed.copy(),  # Filtered image
            'sharpened': rgb_sharpened.copy(),  # Sharpened image
            'filter_settings': {
                'blur_kernel': blur_kernel,
                'smooth_sigma': smooth_sigma
            }
        }
        
        return matched_rgb, material_matrix, quantized_image, debug_data
    
    def _process_pixel_mode(self, rgb_arr, target_h, target_w):
        """
        Pixel art mode image processing
        Direct pixel-level color matching, no smoothing
        """
        print(f"[IMAGE_PROCESSOR] Direct pixel-level matching (Pixel Art mode)...")
        
        flat_rgb = rgb_arr.reshape(-1, 3)
        _, indices = self.kdtree.query(flat_rgb)
        
        matched_rgb = self.lut_rgb[indices].reshape(target_h, target_w, 3)
        material_matrix = self.ref_stacks[indices].reshape(
            target_h, target_w, PrinterConfig.COLOR_LAYERS
        )
        
        print(f"[IMAGE_PROCESSOR] Direct matching complete!")
        
        return matched_rgb, material_matrix, rgb_arr
