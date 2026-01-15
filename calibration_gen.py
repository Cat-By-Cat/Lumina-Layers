"""
Lumina Calibration Generator
----------------------------
A tool to generate multi-material 3D printable color calibration boards
for verifying color mixing behaviors in FDM printing (e.g., HueForge style).

Dependencies: gradio, numpy, trimesh, pillow
Author: [MIN]
License: MIT
"""

import os
import tempfile
from typing import List, Tuple, Dict, Optional, Any

import gradio as gr
import numpy as np
import trimesh
from PIL import Image


# --- Configuration & Constants ---

class PrinterConfig:
    """Configuration for printer physical parameters."""
    LAYER_HEIGHT: float = 0.08  # Z-axis resolution (mm)
    NOZZLE_WIDTH: float = 0.42  # X/Y-axis resolution (mm)
    COLOR_LAYERS: int = 5  # Number of mixing layers (0.4mm total)
    BACKING_MM: float = 1.6  # Thickness of the solid backing (mm)
    SHRINK_OFFSET: float = 0.02  # Tolerance gap to prevent slicer path overlap (mm)


class CalibrationConfig:
    """Configuration for the calibration pattern."""
    GRID_ROWS: int = 32
    GRID_COLS: int = 32
    TOTAL_COLORS: int = 1024  # 4 colors ^ 5 layers
    PREVIEW_COLORS: Dict[int, List[int]] = {
        0: [250, 250, 250, 255],  # White
        1: [220, 20, 60, 255],  # Red
        2: [255, 230, 0, 255],  # Yellow
        3: [0, 100, 240, 255]  # Blue
    }
    SLOT_NAMES: List[str] = ["White", "Red", "Yellow", "Blue"]
    COLOR_MAP: Dict[str, int] = {"White": 0, "Red": 1, "Yellow": 2, "Blue": 3}


# --- Core Logic: Mesh Generation ---

def generate_voxel_mesh(
        voxel_matrix: np.ndarray,
        material_index: int,
        grid_h: int,
        grid_w: int
) -> Optional[trimesh.Trimesh]:
    """
    Converts a specific material layer in the voxel matrix into a 3D mesh.
    Applies shrink tolerances to ensure clean slicer paths.

    Args:
        voxel_matrix: 3D numpy array representing the object (Z, Y, X).
        material_index: The specific material ID to extract.
        grid_h: Height of the voxel grid (Y).
        grid_w: Width of the voxel grid (X).

    Returns:
        A trimesh object or None if no voxels exist for this material.
    """
    scale_x = PrinterConfig.NOZZLE_WIDTH
    scale_y = PrinterConfig.NOZZLE_WIDTH
    scale_z = PrinterConfig.LAYER_HEIGHT
    shrink = PrinterConfig.SHRINK_OFFSET

    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    total_z_layers = voxel_matrix.shape[0]

    for z in range(total_z_layers):
        # Calculate physical Z height for the current layer
        z_bottom = z * scale_z
        z_top = (z + 1) * scale_z

        # Get boolean mask for the current material on this layer
        layer_mask = (voxel_matrix[z] == material_index)
        if not np.any(layer_mask):
            continue

        # Iterate through rows (Y-axis)
        for y in range(grid_h):
            world_y = y * scale_y
            row = layer_mask[y]

            # Optimization: Use run-length encoding logic to find continuous segments
            # Pad to handle edge cases where segment touches borders
            padded_row = np.pad(row, (1, 1), mode='constant')
            diff = np.diff(padded_row.astype(int))

            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for start, end in zip(starts, ends):
                # Apply shrinkage to X and Y coordinates to prevent overlap
                x0 = (start * scale_x) + shrink
                x1 = (end * scale_x) - shrink
                y0 = world_y + shrink
                y1 = world_y + scale_y - shrink

                # Define 8 vertices of the cuboid
                base_idx = len(vertices)
                cube_verts = [
                    [x0, y0, z_bottom], [x1, y0, z_bottom], [x1, y1, z_bottom], [x0, y1, z_bottom],  # Bottom
                    [x0, y0, z_top], [x1, y0, z_top], [x1, y1, z_top], [x0, y1, z_top]  # Top
                ]
                vertices.extend(cube_verts)

                # Define 12 triangles (2 per face * 6 faces)
                # Vertex ordering is crucial for correct normal generation
                cube_faces = [
                    [0, 2, 1], [0, 3, 2],  # Bottom
                    [4, 5, 6], [4, 6, 7],  # Top
                    [0, 1, 5], [0, 5, 4],  # Front
                    [1, 2, 6], [1, 6, 5],  # Right
                    [2, 3, 7], [2, 7, 6],  # Back
                    [3, 0, 4], [3, 4, 7]  # Left
                ]
                faces.extend([[v + base_idx for v in f] for f in cube_faces])

    if not vertices:
        return None

    # Create mesh object and cleanup
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())

    return mesh


# --- Core Logic: Board Layout ---

def construct_calibration_matrix(
        block_size_mm: float,
        gap_mm: float,
        backing_color: str
) -> Tuple[str, Image.Image, str]:
    """
    Generates the full calibration board including color permutations and backing.
    """
    logs = []
    logs.append(f"INFO: Initializing generation for {backing_color} backing...")

    try:
        backing_id = CalibrationConfig.COLOR_MAP[backing_color]
    except KeyError:
        return None, None, "Error: Invalid backing color selected."

    # grid setup
    grid_dim = 32  # 32x32 grid
    padding = 1  # 1 block padding around
    total_w = grid_dim + (padding * 2)
    total_h = grid_dim + (padding * 2)

    # Calculate pixel dimensions based on nozzle width
    pixels_per_block = max(1, int(block_size_mm / PrinterConfig.NOZZLE_WIDTH))
    pixels_gap = max(1, int(gap_mm / PrinterConfig.NOZZLE_WIDTH))

    voxel_w = total_w * (pixels_per_block + pixels_gap)
    voxel_h = total_h * (pixels_per_block + pixels_gap)

    # Calculate Z-axis layers
    backing_layers_count = int(PrinterConfig.BACKING_MM / PrinterConfig.LAYER_HEIGHT)
    total_layers = PrinterConfig.COLOR_LAYERS + backing_layers_count

    # Initialize voxel matrix with backing material
    # Shape: (Z, Y, X)
    full_matrix = np.full(
        (total_layers, voxel_h, voxel_w),
        backing_id,
        dtype=int
    )

    # --- Phase 1: Generate Color Permutations (Top 5 Layers) ---
    # We use a base-4 counter to generate all 1024 combinations
    for i in range(CalibrationConfig.TOTAL_COLORS):
        # Convert index to base-4 digits (representing material IDs)
        digits = []
        temp = i
        for _ in range(5):
            digits.append(temp % 4)
            temp //= 4

        # Face-down logic: High bit is Layer 0 (Bottom/Face)
        # This ensures large visual blocks on the face
        stack = digits[::-1]

        # Calculate grid position (skipping padding border)
        row = (i // grid_dim) + padding
        col = (i % grid_dim) + padding

        # Pixel coordinates
        px_start = col * (pixels_per_block + pixels_gap)
        py_start = row * (pixels_per_block + pixels_gap)

        # Fill the top 5 layers (which are technically bottom 5 in Z-axis)
        for z in range(PrinterConfig.COLOR_LAYERS):
            material = stack[z]
            full_matrix[z,
            py_start: py_start + pixels_per_block,
            px_start: px_start + pixels_per_block] = material

    # --- Phase 2: Corner Markers ---
    # TL:White, TR:Red, BL:Yellow, BR:Blue
    corners = [
        (0, 0, 0),  # TL
        (0, total_w - 1, 1),  # TR
        (total_h - 1, 0, 2),  # BL
        (total_h - 1, total_w - 1, 3)  # BR
    ]

    for r, c, mat_id in corners:
        px_start = c * (pixels_per_block + pixels_gap)
        py_start = r * (pixels_per_block + pixels_gap)

        # Overwrite corners with solid colors
        for z in range(PrinterConfig.COLOR_LAYERS):
            full_matrix[z,
            py_start: py_start + pixels_per_block,
            px_start: px_start + pixels_per_block] = mat_id

    # --- Phase 3: Mesh Export ---
    logs.append("INFO: Generating 3MF geometry...")
    scene = trimesh.Scene()

    for mat_id in range(4):
        mesh = generate_voxel_mesh(full_matrix, mat_id, voxel_h, voxel_w)
        if mesh:
            # Add color metadata for slicers
            rgba = CalibrationConfig.PREVIEW_COLORS[mat_id]
            mesh.visual.face_colors = rgba

            slot_name = CalibrationConfig.SLOT_NAMES[mat_id]
            obj_name = f"Slot_{mat_id + 1}_{slot_name}"

            # Add to scene
            scene.add_geometry(mesh, node_name=obj_name)
            mesh.metadata['name'] = obj_name

    # Export to temp file
    temp_dir = tempfile.gettempdir()
    output_filename = "Lumina_Calibration_SolidBacking.3mf"
    output_path = os.path.join(temp_dir, output_filename)
    scene.export(output_path)

    logs.append("SUCCESS: Model generated successfully.")

    # --- Phase 4: Preview Generation ---
    # Generate a preview image from Layer 0 (Face)
    bottom_layer = full_matrix[0].astype(np.uint8)
    preview_arr = np.zeros((voxel_h, voxel_w, 3), dtype=np.uint8)

    for mat_id, rgba in CalibrationConfig.PREVIEW_COLORS.items():
        mask = (bottom_layer == mat_id)
        preview_arr[mask] = rgba[:3]  # Use only RGB

    preview_img = Image.fromarray(preview_arr)

    return output_path, preview_img, "\n".join(logs)


# --- User Interface ---

def create_ui():
    with gr.Blocks(title="Lumina Calibration Generator") as app:
        gr.Markdown(
            """
            # 🎨 Lumina Calibration Generator
            **Solid Backing Edition**

            Generates a 1024-color calibration board with a solid 1.6mm backing plate. 
            Designed for "Face-Down" printing to ensure color purity and handling durability.

            **Slicer Settings:**
            - **Layer Height:** 0.08mm (Crucial)
            - **First Layer:** 0.08mm
            - **Colors:** Slot 1=White, 2=Red, 3=Yellow, 4=Blue
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### ⚙️ Parameters")
                    size_slider = gr.Slider(
                        minimum=3, maximum=10, value=3, step=1,
                        label="Block Size (mm)"
                    )
                    gap_slider = gr.Slider(
                        minimum=0.5, maximum=2.0, value=0.5,
                        label="Grid Gap (mm)"
                    )
                    grid_col = gr.Dropdown(
                        choices=["White", "Blue", "Red", "Yellow"],
                        value="White",
                        label="Backing Color"
                    )

                generate_btn = gr.Button("🚀 Generate 3MF", variant="primary")

                with gr.Accordion("Logs", open=True):
                    log_output = gr.Textbox(label="System Output", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### 👁️ Preview (Face View)")
                preview_image = gr.Image(label="Layer 0 Preview", show_label=False)
                file_output = gr.File(label="Download Model")

        # Event Bindings
        generate_btn.click(
            fn=construct_calibration_matrix,
            inputs=[size_slider, gap_slider, grid_col],
            outputs=[file_output, preview_image, log_output]
        )

    return app


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(inbrowser=True)
