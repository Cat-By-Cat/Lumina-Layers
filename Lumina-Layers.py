"""
Lumina Generation Engine
------------------------
Part of the Lumina-Layers Ecosystem.

This engine converts 2D images into multi-layered 3D models optimized for
FDM color mixing (HueForge-style).

Key Features:
- Closed-Loop Color Matching: Uses real calibration data (.npy LUT)
- Adaptive Integer Geometry: Generates slice-friendly meshes (no overlap)
- Transparency Handling: Supports PNG alpha channels & auto-background removal
- White Backing Correction: Forces spacer layers to be white for better vibrancy
- Live Color Preview: WYSIWYG preview of the final print colors

Author: [MIN]
License: CC BY-NC-SA 4.0
"""

import gradio as gr
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
import trimesh
import os
import tempfile

# --- 1. Configuration ---

FILAMENT_MAP = {
    0: {'name': 'White', 'rgb': [250, 250, 250]},
    1: {'name': 'Red', 'rgb': [220, 20, 60]},
    2: {'name': 'Yellow', 'rgb': [255, 230, 0]},
    3: {'name': 'Blue', 'rgb': [0, 100, 240]}
}

CONFIG = {
    'layer_height': 0.08,
    'nozzle_width': 0.42,
    'layers_per_stack': 5,
    'alpha_threshold': 10,
}


# --- 2. Data Loading (LUT) ---

def load_calibrated_lut(npy_path):
    """Loads the .npy calibration file and filters valid color stacks."""
    try:
        lut_grid = np.load(npy_path)
        measured_colors = lut_grid.reshape(-1, 3)
    except:
        return None, None, "❌ LUT File Corrupted"

    valid_rgb = []
    valid_stacks = []
    dropped_count = 0

    # Filter logic (legacy blue-base rejection, kept for safety)
    base_blue = np.array([30, 100, 200])

    for i in range(1024):
        digits = []
        temp = i
        for _ in range(5):
            digits.append(temp % 4)
            temp //= 4
        stack = digits[::-1]

        real_rgb = measured_colors[i]

        dist_to_base = np.linalg.norm(real_rgb - base_blue)
        has_blue_in_stack = (3 in stack)

        if dist_to_base < 60 and not has_blue_in_stack:
            dropped_count += 1
            continue

        valid_rgb.append(real_rgb)
        valid_stacks.append(stack)

    return np.array(valid_rgb), np.array(valid_stacks), f"✅ LUT Loaded ({dropped_count} bad points dropped)"


# --- 3. Geometry Generation ---

def create_integer_slab_mesh(voxel_matrix, mat_id, height):
    """
    Generates a 3D mesh from voxel data using the 'Integer Slab' algorithm.
    Optimized for slicers by applying micro-shrinkage to avoid overlapping paths.
    """
    vertices = []
    faces = []

    scale_x = 1.0
    scale_y = 1.0
    scale_z = 1.0

    # Micro-shrinkage to prevent line width overlap in slicers
    # 0.05 units approx 0.021mm
    shrink = 0.05

    total_z = voxel_matrix.shape[0]

    for z in range(total_z):
        layer_z_bottom = z * scale_z
        layer_z_top = (z + 1) * scale_z

        mask = (voxel_matrix[z] == mat_id)
        if not np.any(mask): continue

        for y in range(height):
            world_y = (height - 1 - y) * scale_y
            row = mask[y]

            # Run-Length Encoding for efficient geometry
            padded = np.pad(row, (1, 1), mode='constant')
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for start, end in zip(starts, ends):
                x0_raw = start * scale_x
                x1_raw = end * scale_x
                y0_raw = world_y
                y1_raw = world_y + scale_y

                # Apply shrinkage
                x0 = x0_raw + shrink
                x1 = x1_raw - shrink
                y0 = y0_raw + shrink
                y1 = y1_raw - shrink
                z0 = layer_z_bottom
                z1 = layer_z_top

                base_idx = len(vertices)
                vs = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
                vertices.extend(vs)
                new_faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                             [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                             [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
                faces.extend([[v + base_idx for v in f] for f in new_faces])

    if not vertices: return None
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    return mesh


# --- 4. Main Processing Pipeline ---

def process_engine(image_path, lut_path, target_width_mm, spacer_thick,
                   structure_mode, auto_bg, bg_tol):
    status_log = []

    if image_path is None: return None, None, "❌ Please upload an image."

    if lut_path is not None:
        status_log.append("📂 Loading LUT...")
        lut_rgb, ref_stacks, msg = load_calibrated_lut(lut_path.name)
        status_log.append(msg)
        if lut_rgb is None or len(lut_rgb) == 0:
            return None, None, "❌ Failed to load LUT"
        tree = KDTree(lut_rgb)
    else:
        return None, None, "⚠️ Please upload a .npy calibration file!"

    # 1. Image Preprocessing
    img = Image.open(image_path).convert('RGBA')
    target_pixel_width = int(target_width_mm / CONFIG['nozzle_width'])
    aspect_ratio = img.height / img.width
    target_pixel_height = int(target_pixel_width * aspect_ratio)

    # Nearest Neighbor scaling for pixel-perfect art
    img = img.resize((target_pixel_width, target_pixel_height), Image.Resampling.NEAREST)
    img_arr = np.array(img)
    rgb_arr = img_arr[:, :, :3]
    alpha_arr = img_arr[:, :, 3]

    status_log.append(f"🖼️ Resolution: {target_pixel_width}x{target_pixel_height} px")

    # 2. Color Matching
    flat_rgb = rgb_arr.reshape(-1, 3)
    status_log.append("🔍 Matching Colors...")
    _, indices = tree.query(flat_rgb)

    # Retrieve matched data
    matched_rgb = lut_rgb[indices].reshape(target_pixel_height, target_pixel_width, 3)
    best_stacks = ref_stacks[indices].reshape(target_pixel_height, target_pixel_width, CONFIG['layers_per_stack'])

    # 3. Transparency & Background Handling
    mask_transparent = alpha_arr < CONFIG['alpha_threshold']

    if auto_bg:
        # Simple color-based background removal
        bg_color = rgb_arr[0, 0]
        diff = np.sum(np.abs(rgb_arr - bg_color), axis=-1)
        mask_bg = diff < bg_tol
        mask_transparent = np.logical_or(mask_transparent, mask_bg)

    # Mask out transparent areas in stack data
    best_stacks[mask_transparent] = -1

    # Generate Color Preview (RGBA)
    status_log.append("🎨 Generating Preview...")
    preview_rgba = np.zeros((target_pixel_height, target_pixel_width, 4), dtype=np.uint8)
    mask_solid = ~mask_transparent
    preview_rgba[mask_solid, :3] = matched_rgb[mask_solid]
    preview_rgba[mask_solid, 3] = 255
    preview_img = Image.fromarray(preview_rgba, mode='RGBA')

    # 4. Voxel Construction
    bottom_voxels = np.transpose(best_stacks, (2, 0, 1))
    spacer_layers = max(1, int(round(spacer_thick / CONFIG['layer_height'])))

    # Force Spacer/Backing to be White (ID=0)
    spacer_id = 0

    if "Double" in structure_mode:
        # Double-sided (Keychain style)
        top_voxels = np.transpose(best_stacks[..., ::-1], (2, 0, 1))
        total_layers = 5 + spacer_layers + 5
        full_matrix = np.full((total_layers, target_pixel_height, target_pixel_width), -1, dtype=int)
        full_matrix[0:5] = bottom_voxels

        single_spacer = np.full((target_pixel_height, target_pixel_width), -1, dtype=int)
        single_spacer[~mask_transparent] = spacer_id
        for z in range(5, 5 + spacer_layers): full_matrix[z] = single_spacer

        full_matrix[5 + spacer_layers:] = top_voxels

    else:
        # Single-sided (Standard Relief)
        total_layers = 5 + spacer_layers
        full_matrix = np.full((total_layers, target_pixel_height, target_pixel_width), -1, dtype=int)
        full_matrix[0:5] = bottom_voxels

        single_spacer = np.full((target_pixel_height, target_pixel_width), -1, dtype=int)
        single_spacer[~mask_transparent] = spacer_id
        for z in range(5, total_layers): full_matrix[z] = single_spacer

    # 5. Mesh Generation
    status_log.append("⚙️ Building 3D Mesh...")
    scene = trimesh.Scene()

    transform_matrix = np.eye(4)
    transform_matrix[0, 0] = CONFIG['nozzle_width']
    transform_matrix[1, 1] = CONFIG['nozzle_width']
    transform_matrix[2, 2] = CONFIG['layer_height']

    disp_colors = {
        0: [250, 250, 250, 255],
        1: [220, 20, 60, 255],
        2: [255, 230, 0, 255],
        3: [0, 100, 240, 255]
    }
    slot_names = ["White", "Red", "Yellow", "Blue"]

    for mat_id in range(4):
        mesh = create_integer_slab_mesh(full_matrix, mat_id, target_pixel_height)
        if mesh:
            mesh.apply_transform(transform_matrix)
            mesh.visual.face_colors = disp_colors[mat_id]
            mesh.metadata['name'] = f"Slot_{mat_id + 1}_{slot_names[mat_id]}"
            scene.add_geometry(mesh, node_name=f"Slot_{mat_id + 1}")

    # Export
    temp_dir = tempfile.gettempdir()
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(temp_dir, f"{base_name}_Lumina.3mf")
    scene.export(out_path)
    status_log.append("✅ Done!")

    return out_path, preview_img, "\n".join(status_log)


# --- 5. UI Layout ---

with gr.Blocks(title="Lumina Engine") as app:
    gr.Markdown("# 💎 Lumina Generation Engine")
    gr.Markdown("Closed-loop multi-color printing. Upload your LUT and Image to begin.")

    with gr.Row():
        # --- Left Column: Inputs ---
        with gr.Column():
            lut_file = gr.File(label="1. Calibration Data (.npy)", file_types=['.npy'])
            input_file = gr.Image(label="2. Input Image (Pixel Art)", type="filepath")

            structure = gr.Radio(
                ["Double Sided (Keychain)", "Single Sided (Relief)"],
                value="Double Sided (Keychain)",
                label="Structure Type"
            )

            with gr.Group():
                auto_bg = gr.Checkbox(label="Auto Background Removal", value=True)
                tol = gr.Slider(0, 150, 40, label="Tolerance")

            width = gr.Slider(20, 150, 60, label="Target Width (mm)")
            thick = gr.Slider(0.2, 2.0, 0.64, step=0.08, label="Backing Thickness (mm)")

            btn = gr.Button("🚀 Generate Model", variant="primary")

        # --- Right Column: Outputs ---
        with gr.Column():
            mask_view = gr.Image(label="Color Preview (What you get)", type="pil")
            out_file = gr.File(label="Download 3MF")
            log = gr.Textbox(label="Log", lines=3)

    btn.click(process_engine,
              inputs=[input_file, lut_file, width, thick, structure, auto_bg, tol],
              outputs=[out_file, mask_view, log])

if __name__ == "__main__":
    app.launch(inbrowser=True)
