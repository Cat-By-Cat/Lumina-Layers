"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         CHROMA K/S ENGINE MODULE                              ║
║                    Kubelka-Munk 物理光学引擎                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

This module contains the Chroma K/S physics-based color calculation engine,
adapted from ChromaStack project.

Core Components:
- physics.py: K-M theory calculations and LUT generation
- calibration.py: Step card generation
- calibration_ks.py: K/S parameter calculation from photos
- converter.py: Image-to-3D conversion with dynamic color matching
- filament_db.py: Filament database management
- ui_components.py: Gradio UI components for K/S workflow
"""

__version__ = "1.0.0"
__author__ = "Lumina Studio Team"

# Version check
try:
    from .physics import VirtualPhysics, rgb_to_lab
    from .calibration import generate_step_card
    from .calibration_ks import (
        process_calibration_image,
        calculate_and_plot_km,
        sample_step_card_colors,
        apply_perspective_transform,
        auto_white_balance_by_paper
    )
    
    __all__ = [
        'VirtualPhysics',
        'rgb_to_lab',
        'generate_step_card',
        'process_calibration_image',
        'calculate_and_plot_km',
        'sample_step_card_colors',
        'apply_perspective_transform',
        'auto_white_balance_by_paper'
    ]
    
    KS_ENGINE_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️  K/S Engine components not fully available: {e}")
    KS_ENGINE_AVAILABLE = False
    __all__ = []
