"""
Lumina Studio - UI Layout (Refactored with i18n)
UI layout definition - Refactored version with language switching support
"""

import gradio as gr

from core.i18n import I18n
from config import ColorSystem
from utils import Stats, LUTManager
from core.calibration import generate_calibration_board
from core.extractor import (
    rotate_image,
    draw_corner_points,
    run_extraction,
    probe_lut_cell,
    manual_fix_cell,
    generate_simulated_reference
)
from core.converter import (
    generate_preview_cached,
    render_preview,
    on_preview_click,
    update_preview_with_loop,
    on_remove_loop,
    generate_final_model
)
from .styles import CUSTOM_CSS
from .callbacks import (
    get_first_hint,
    get_next_hint,
    on_extractor_upload,
    on_extractor_mode_change,
    on_extractor_rotate,
    on_extractor_click,
    on_extractor_clear,
    on_lut_select,
    on_lut_upload_save
)


def create_app():
    """Create Gradio application interface (with language switching support)"""
    with gr.Blocks(title="Lumina Studio") as app:
        
        # ==================== Language State ====================
        lang_state = gr.State(value="zh")  # Default Chinese
        
        # ==================== Header ====================
        with gr.Row():
            with gr.Column(scale=10):
                app_title_html = gr.HTML(
                    value=_get_header_html("zh"),
                    elem_id="app-header"
                )
            with gr.Column(scale=1, min_width=120):
                lang_btn = gr.Button(
                    value="🌐 English",
                    size="sm",
                    elem_id="lang-btn"
                )
        
        # ==================== Stats Bar ====================
        stats = Stats.get_all()
        stats_html = gr.HTML(
            value=_get_stats_html("zh", stats),
            elem_id="stats-bar"
        )
        
        # ==================== Main Tabs ====================
        # Create tab container (to store tab components)
        tab_components = {}
        
        with gr.Tabs() as tabs:
            # Store all components that need updating
            components = {}
            
            # Tab 1: Image Converter
            with gr.TabItem(label=I18n.get('tab_converter', "zh"), id=0) as tab_conv:
                conv_components = create_converter_tab_content("zh")
                components.update(conv_components)
            tab_components['tab_converter'] = tab_conv
            
            # Tab 2: Calibration Board Generator
            with gr.TabItem(label=I18n.get('tab_calibration', "zh"), id=1) as tab_cal:
                cal_components = create_calibration_tab_content("zh")
                components.update(cal_components)
            tab_components['tab_calibration'] = tab_cal
            
            # Tab 3: Color Extractor
            with gr.TabItem(label=I18n.get('tab_extractor', "zh"), id=2) as tab_ext:
                ext_components = create_extractor_tab_content("zh")
                components.update(ext_components)
            tab_components['tab_extractor'] = tab_ext
            
            # Tab 4: About
            with gr.TabItem(label=I18n.get('tab_about', "zh"), id=3) as tab_about:
                about_components = create_about_tab_content("zh")
                components.update(about_components)
            tab_components['tab_about'] = tab_about
        
        # ==================== Footer ====================
        footer_html = gr.HTML(
            value=_get_footer_html("zh"),
            elem_id="footer"
        )
        
        # ==================== Language Switching Logic ====================
        def change_language(current_lang):
            """Switch language"""
            new_lang = "en" if current_lang == "zh" else "zh"
            
            # Prepare all updates
            updates = []
            
            # 1. Update language button
            updates.append(gr.update(value=I18n.get('lang_btn_zh' if new_lang == "zh" else 'lang_btn_en', new_lang)))
            
            # 2. Update header
            updates.append(gr.update(value=_get_header_html(new_lang)))
            
            # 3. Update stats bar
            stats = Stats.get_all()
            updates.append(gr.update(value=_get_stats_html(new_lang, stats)))
            
            # 4. Update tab titles
            updates.append(gr.update(label=I18n.get('tab_converter', new_lang)))
            updates.append(gr.update(label=I18n.get('tab_calibration', new_lang)))
            updates.append(gr.update(label=I18n.get('tab_extractor', new_lang)))
            updates.append(gr.update(label=I18n.get('tab_about', new_lang)))
            
            # 5. Update all components
            updates.extend(_get_all_component_updates(new_lang, components))
            
            # 6. Update footer
            updates.append(gr.update(value=_get_footer_html(new_lang)))
            
            # 7. Update language state
            updates.append(new_lang)
            
            return updates
            
            # 5. Update all components
            updates.extend(_get_all_component_updates(new_lang, components))
            
            # 6. Update footer
            updates.append(gr.update(value=_get_footer_html(new_lang)))
            
            # 7. Update language state
            updates.append(new_lang)
            
            return updates
        
        # Bind language switching event
        output_list = [
            lang_btn,
            app_title_html,
            stats_html,
            tab_components['tab_converter'],
            tab_components['tab_calibration'],
            tab_components['tab_extractor'],
            tab_components['tab_about'],
        ]
        
        # Add all components to output list
        output_list.extend(_get_component_list(components))
        
        # Add footer and language state
        output_list.extend([footer_html, lang_state])
        
        lang_btn.click(
            change_language,
            inputs=[lang_state],
            outputs=output_list
        )
    
    return app


# ==================== Helper Functions ====================

def _get_header_html(lang: str) -> str:
    """Generate header HTML"""
    return f"""
    <div class="header-banner">
        <h1>{I18n.get('app_title', lang)}</h1>
        <p>{I18n.get('app_subtitle', lang)}</p>
    </div>
    """


def _get_stats_html(lang: str, stats: dict) -> str:
    """Generate stats bar HTML"""
    return f"""
    <div class="stats-bar">
        {I18n.get('stats_total', lang)}: 
        <strong>{stats.get('calibrations', 0)}</strong> {I18n.get('stats_calibrations', lang)} | 
        <strong>{stats.get('extractions', 0)}</strong> {I18n.get('stats_extractions', lang)} | 
        <strong>{stats.get('conversions', 0)}</strong> {I18n.get('stats_conversions', lang)}
    </div>
    """


def _get_footer_html(lang: str) -> str:
    """Generate footer HTML"""
    return f"""
    <div class="footer">
        <p>{I18n.get('footer_tip', lang)}</p>
    </div>
    """


def _get_all_component_updates(lang: str, components: dict) -> list:
    """
    Get update list for all components
    
    Args:
        lang: Target language
        components: Component dictionary
    
    Returns:
        list: gr.update() list
    """
    updates = []
    
    # Update in order of component dictionary
    for key, component in components.items():
        if key.startswith('md_'):
            # Markdown component
            updates.append(gr.update(value=I18n.get(key[3:], lang)))
        elif key.startswith('lbl_'):
            # Label component
            updates.append(gr.update(label=I18n.get(key[4:], lang)))
        elif key.startswith('btn_'):
            # Button component
            updates.append(gr.update(value=I18n.get(key[4:], lang)))
        elif key.startswith('radio_'):
            # Radio component - need to update choices
            choice_key = key[6:]
            if choice_key == 'conv_color_mode' or choice_key == 'cal_color_mode' or choice_key == 'ext_color_mode':
                updates.append(gr.update(
                    label=I18n.get(choice_key, lang),
                    choices=[
                        I18n.get('conv_color_mode_cmyw', lang),
                        I18n.get('conv_color_mode_rybw', lang)
                    ]
                ))
            elif choice_key == 'conv_structure':
                updates.append(gr.update(
                    label=I18n.get(choice_key, lang),
                    choices=[
                        I18n.get('conv_structure_double', lang),
                        I18n.get('conv_structure_single', lang)
                    ]
                ))
            elif choice_key == 'conv_modeling_mode':
                updates.append(gr.update(
                    label=I18n.get(choice_key, lang),
                    info=I18n.get('conv_modeling_mode_info', lang),
                    choices=[
                        I18n.get('conv_modeling_mode_hifi', lang),
                        I18n.get('conv_modeling_mode_pixel', lang)
                    ]
                ))
        elif key.startswith('slider_'):
            # Slider component
            slider_key = key[7:]
            updates.append(gr.update(label=I18n.get(slider_key, lang)))
        elif key.startswith('checkbox_'):
            # Checkbox component
            checkbox_key = key[9:]
            info_key = checkbox_key + '_info'
            if info_key in I18n.TEXTS:
                updates.append(gr.update(
                    label=I18n.get(checkbox_key, lang),
                    info=I18n.get(info_key, lang)
                ))
            else:
                updates.append(gr.update(label=I18n.get(checkbox_key, lang)))
        elif key.startswith('dropdown_'):
            # Dropdown component
            dropdown_key = key[9:]
            info_key = dropdown_key + '_info'
            if info_key in I18n.TEXTS:
                updates.append(gr.update(
                    label=I18n.get(dropdown_key, lang),
                    info=I18n.get(info_key, lang)
                ))
            else:
                updates.append(gr.update(label=I18n.get(dropdown_key, lang)))
        elif key.startswith('image_'):
            # Image component
            image_key = key[6:]
            updates.append(gr.update(label=I18n.get(image_key, lang)))
        elif key.startswith('file_'):
            # File component
            file_key = key[5:]
            updates.append(gr.update(label=I18n.get(file_key, lang)))
        elif key.startswith('textbox_'):
            # Textbox component
            textbox_key = key[8:]
            updates.append(gr.update(label=I18n.get(textbox_key, lang)))
        elif key.startswith('html_'):
            # HTML component
            html_key = key[5:]
            updates.append(gr.update(value=I18n.get(html_key, lang)))
        else:
            # Other components, try to update directly
            updates.append(gr.update())
    
    return updates


def _get_component_list(components: dict) -> list:
    """
    Get component list (for output)
    
    Args:
        components: Component dictionary
    
    Returns:
        list: Component object list
    """
    return list(components.values())



# ==================== Tab Creation Functions ====================

def create_converter_tab_content(lang: str) -> dict:
    """
    Create image converter tab content
    
    Args:
    lang: Initial language
    
    Returns:
    dict: Component dictionary {key: component}
    """
    components = {}
    
    # Title and description
    components['md_conv_title'] = gr.Markdown(I18n.get('conv_title', lang))
    components['md_conv_desc'] = gr.Markdown(I18n.get('conv_desc', lang))
    
    # State variables
    conv_loop_pos = gr.State(None)
    conv_preview_cache = gr.State(None)
    
    with gr.Row():
        # ========== Left: Input and Parameters ==========
        with gr.Column(scale=1):
            components['md_conv_input_section'] = gr.Markdown(I18n.get('conv_input_section', lang))
                
            # LUT selector
            with gr.Group():
                components['md_conv_lut_title'] = gr.Markdown(I18n.get('conv_lut_title', lang))
                    
                components['dropdown_conv_lut_dropdown'] = gr.Dropdown(
                    choices=LUTManager.get_lut_choices(),
                    label=I18n.get('conv_lut_dropdown', lang),
                    value=None,
                    interactive=True,
                    info=I18n.get('conv_lut_info', lang)
                )
                    
                conv_lut_upload = gr.File(
                    label="",
                    show_label=False,
                    file_types=['.npy'],
                    height=60,
                    elem_classes=["micro-upload"]
                )
                    
                components['md_conv_lut_status'] = gr.Markdown(
                    value=I18n.get('conv_lut_status_default', lang),
                    visible=True
                )
                
            conv_lut_path = gr.State(None)
                
            # Input image
            components['image_conv_image_label'] = gr.Image(
                label=I18n.get('conv_image_label', lang),
                type="filepath",
                image_mode="RGBA"  # Force RGBA mode to preserve alpha channel
            )
                
            # Parameter settings
            components['md_conv_params_section'] = gr.Markdown(I18n.get('conv_params_section', lang))
                
            components['radio_conv_color_mode'] = gr.Radio(
                choices=[
                    I18n.get('conv_color_mode_cmyw', lang),
                    I18n.get('conv_color_mode_rybw', lang)
                ],
                value=I18n.get('conv_color_mode_rybw', lang),
                label=I18n.get('conv_color_mode', lang)
            )
                
            components['radio_conv_structure'] = gr.Radio(
                choices=[
                    I18n.get('conv_structure_double', lang),
                    I18n.get('conv_structure_single', lang)
                ],
                value=I18n.get('conv_structure_double', lang),
                label=I18n.get('conv_structure', lang)
            )
                
            components['radio_conv_modeling_mode'] = gr.Radio(
                choices=[
                    I18n.get('conv_modeling_mode_hifi', lang),
                    I18n.get('conv_modeling_mode_pixel', lang)
                ],
                value=I18n.get('conv_modeling_mode_hifi', lang),
                label=I18n.get('conv_modeling_mode', lang),
                info=I18n.get('conv_modeling_mode_info', lang)
            )
                
            components['slider_conv_quantize_colors'] = gr.Slider(
                minimum=8, maximum=256, step=8, value=64,
                label=I18n.get('conv_quantize_colors', lang),
                info=I18n.get('conv_quantize_info', lang)
            )
                
            components['checkbox_conv_auto_bg'] = gr.Checkbox(
                label=I18n.get('conv_auto_bg', lang),
                value=True,
                info=I18n.get('conv_auto_bg_info', lang)
            )
                
            components['slider_conv_tolerance'] = gr.Slider(
                0, 150, 40,
                label=I18n.get('conv_tolerance', lang),
                info=I18n.get('conv_tolerance_info', lang)
            )
                
            components['slider_conv_width'] = gr.Slider(
                20, 400, 60,
                label=I18n.get('conv_width', lang)
            )
                
            components['slider_conv_thickness'] = gr.Slider(
                0.2, 3.5, 1.2, step=0.08,
                label=I18n.get('conv_thickness', lang)
            )
                
            components['btn_conv_preview_btn'] = gr.Button(
                I18n.get('conv_preview_btn', lang),
                variant="secondary",
                size="lg"
            )
            
        # ========== Middle: Preview Edit Area ==========
        with gr.Column(scale=2):
            components['md_conv_preview_section'] = gr.Markdown(
                I18n.get('conv_preview_section', lang)
            )
                
            conv_preview = gr.Image(
                label="",
                type="numpy",
                height=500,
                interactive=False,
                show_label=False
            )
                
            # Loop settings
            with gr.Group():
                components['md_conv_loop_section'] = gr.Markdown(
                    I18n.get('conv_loop_section', lang)
                )
                    
                with gr.Row():
                    components['checkbox_conv_loop_enable'] = gr.Checkbox(
                        label=I18n.get('conv_loop_enable', lang),
                        value=False
                    )
                    components['btn_conv_loop_remove'] = gr.Button(
                        I18n.get('conv_loop_remove', lang),
                        size="sm"
                    )
                    
                with gr.Row():
                    components['slider_conv_loop_width'] = gr.Slider(
                        2, 10, 4, step=0.5,
                        label=I18n.get('conv_loop_width', lang)
                    )
                    components['slider_conv_loop_length'] = gr.Slider(
                        4, 15, 8, step=0.5,
                        label=I18n.get('conv_loop_length', lang)
                    )
                    components['slider_conv_loop_hole'] = gr.Slider(
                        1, 5, 2.5, step=0.25,
                        label=I18n.get('conv_loop_hole', lang)
                    )
                    
                with gr.Row():
                    components['slider_conv_loop_angle'] = gr.Slider(
                        -180, 180, 0, step=5,
                        label=I18n.get('conv_loop_angle', lang)
                    )
                    components['textbox_conv_loop_info'] = gr.Textbox(
                        label=I18n.get('conv_loop_info', lang),
                        interactive=False,
                        scale=2
                    )
                
            components['textbox_conv_status'] = gr.Textbox(
                label=I18n.get('conv_status', lang),
                lines=6,
                interactive=False,
                max_lines=10,
                show_label=True
            )
            
        # ========== Right: Output ==========
        with gr.Column(scale=1):
            components['btn_conv_generate_btn'] = gr.Button(
                I18n.get('conv_generate_btn', lang),
                variant="primary",
                size="lg"
            )
                
            components['md_conv_3d_preview'] = gr.Markdown(
                I18n.get('conv_3d_preview', lang)
            )
                
            conv_3d_preview = gr.Model3D(
                label="3D",
                clear_color=[0.9, 0.9, 0.9, 1.0],
                height=280
            )
                
            components['md_conv_download_section'] = gr.Markdown(
                I18n.get('conv_download_section', lang)
            )
                
            components['file_conv_download_file'] = gr.File(
                label=I18n.get('conv_download_file', lang)
            )
    
    # ========== Event Binding ==========
    # LUT selection event
    components['dropdown_conv_lut_dropdown'].change(
            on_lut_select,
            inputs=[components['dropdown_conv_lut_dropdown']],
            outputs=[conv_lut_path, components['md_conv_lut_status']]
    )
    
    # LUT upload event
    conv_lut_upload.upload(
            on_lut_upload_save,
            inputs=[conv_lut_upload],
            outputs=[components['dropdown_conv_lut_dropdown'], components['md_conv_lut_status']]
    )
    
    # Generate preview
    components['btn_conv_preview_btn'].click(
            generate_preview_cached,
            inputs=[
                components['image_conv_image_label'],
                conv_lut_path,
                components['slider_conv_width'],
                components['checkbox_conv_auto_bg'],
                components['slider_conv_tolerance'],
                components['radio_conv_color_mode']
            ],
            outputs=[conv_preview, conv_preview_cache, components['textbox_conv_status']]
    )
    
    # Click preview image to place loop
    conv_preview.select(
            on_preview_click,
            inputs=[conv_preview_cache, conv_loop_pos],
            outputs=[conv_loop_pos, components['checkbox_conv_loop_enable'], components['textbox_conv_loop_info']]
    ).then(
            update_preview_with_loop,
            inputs=[
                conv_preview_cache, conv_loop_pos, components['checkbox_conv_loop_enable'],
                components['slider_conv_loop_width'], components['slider_conv_loop_length'],
                components['slider_conv_loop_hole'], components['slider_conv_loop_angle']
            ],
            outputs=[conv_preview]
    )
    
    # Remove loop
    components['btn_conv_loop_remove'].click(
            on_remove_loop,
            outputs=[conv_loop_pos, components['checkbox_conv_loop_enable'], 
                    components['slider_conv_loop_angle'], components['textbox_conv_loop_info']]
    ).then(
            update_preview_with_loop,
            inputs=[
                conv_preview_cache, conv_loop_pos, components['checkbox_conv_loop_enable'],
                components['slider_conv_loop_width'], components['slider_conv_loop_length'],
                components['slider_conv_loop_hole'], components['slider_conv_loop_angle']
            ],
            outputs=[conv_preview]
    )
    
    # Update preview in real-time when loop parameters change
    loop_params = [
            components['slider_conv_loop_width'],
            components['slider_conv_loop_length'],
            components['slider_conv_loop_hole'],
            components['slider_conv_loop_angle']
    ]
    for param in loop_params:
            param.change(
                update_preview_with_loop,
                inputs=[
                    conv_preview_cache, conv_loop_pos, components['checkbox_conv_loop_enable'],
                    components['slider_conv_loop_width'], components['slider_conv_loop_length'],
                    components['slider_conv_loop_hole'], components['slider_conv_loop_angle']
                ],
                outputs=[conv_preview]
            )
    
    # Generate final model
    components['btn_conv_generate_btn'].click(
            generate_final_model,
            inputs=[
                components['image_conv_image_label'],
                conv_lut_path,
                components['slider_conv_width'],
                components['slider_conv_thickness'],
                components['radio_conv_structure'],
                components['checkbox_conv_auto_bg'],
                components['slider_conv_tolerance'],
                components['radio_conv_color_mode'],
                components['checkbox_conv_loop_enable'],
                components['slider_conv_loop_width'],
                components['slider_conv_loop_length'],
                components['slider_conv_loop_hole'],
                conv_loop_pos,
                components['radio_conv_modeling_mode'],
                components['slider_conv_quantize_colors']
            ],
            outputs=[
                components['file_conv_download_file'],
                conv_3d_preview,
                conv_preview,
                components['textbox_conv_status']
            ]
    )
    
    return components



def create_calibration_tab_content(lang: str) -> dict:
    """Create calibration board generation tab content"""
    components = {}
    
    components['md_cal_title'] = gr.Markdown(I18n.get('cal_title', lang))
    components['md_cal_desc'] = gr.Markdown(I18n.get('cal_desc', lang))
    
    with gr.Row():
        with gr.Column(scale=1):
            components['md_cal_params'] = gr.Markdown(I18n.get('cal_params', lang))
                
            components['radio_cal_color_mode'] = gr.Radio(
                choices=[
                    I18n.get('conv_color_mode_cmyw', lang),
                    I18n.get('conv_color_mode_rybw', lang)
                ],
                value=I18n.get('conv_color_mode_rybw', lang),
                label=I18n.get('cal_color_mode', lang)
            )
                
            components['slider_cal_block_size'] = gr.Slider(
                3, 10, 5, step=1,
                label=I18n.get('cal_block_size', lang)
            )
                
            components['slider_cal_gap'] = gr.Slider(
                0.4, 2.0, 0.82, step=0.02,
                label=I18n.get('cal_gap', lang)
            )
                
            components['dropdown_cal_backing'] = gr.Dropdown(
                choices=["White", "Cyan", "Magenta", "Yellow", "Red", "Blue"],
                value="White",
                label=I18n.get('cal_backing', lang)
            )
                
            components['btn_cal_generate_btn'] = gr.Button(
                I18n.get('cal_generate_btn', lang),
                variant="primary",
                elem_classes=["primary-btn"]
            )
                
            components['textbox_cal_status'] = gr.Textbox(
                label=I18n.get('cal_status', lang),
                interactive=False
            )
            
        with gr.Column(scale=1):
            components['md_cal_preview'] = gr.Markdown(I18n.get('cal_preview', lang))
                
            cal_preview = gr.Image(
                label="Calibration Preview",
                show_label=False
            )
                
            components['file_cal_download'] = gr.File(
                label=I18n.get('cal_download', lang)
            )
    
    # Event binding
    components['btn_cal_generate_btn'].click(
            generate_calibration_board,
            inputs=[
                components['radio_cal_color_mode'],
                components['slider_cal_block_size'],
                components['slider_cal_gap'],
                components['dropdown_cal_backing']
            ],
            outputs=[
                components['file_cal_download'],
                cal_preview,
                components['textbox_cal_status']
            ]
    )
    
    return components


def create_extractor_tab_content(lang: str) -> dict:
    """Create color extractor tab content"""
    components = {}
    
    components['md_ext_title'] = gr.Markdown(I18n.get('ext_title', lang))
    components['md_ext_desc'] = gr.Markdown(I18n.get('ext_desc', lang))
    
    ext_state_img = gr.State(None)
    ext_state_pts = gr.State([])
    ext_curr_coord = gr.State(None)
    ref_img = generate_simulated_reference()
    
    with gr.Row():
        with gr.Column(scale=1):
            components['md_ext_upload_section'] = gr.Markdown(
                I18n.get('ext_upload_section', lang)
            )
                
            components['radio_ext_color_mode'] = gr.Radio(
                choices=[
                    I18n.get('conv_color_mode_cmyw', lang),
                    I18n.get('conv_color_mode_rybw', lang)
                ],
                value=I18n.get('conv_color_mode_rybw', lang),
                label=I18n.get('ext_color_mode', lang)
            )
                
            ext_img_in = gr.Image(
                label=I18n.get('ext_photo', lang),
                type="numpy",
                interactive=True
            )
                
            with gr.Row():
                components['btn_ext_rotate_btn'] = gr.Button(
                    I18n.get('ext_rotate_btn', lang)
                )
                components['btn_ext_reset_btn'] = gr.Button(
                    I18n.get('ext_reset_btn', lang)
                )
                
            components['md_ext_correction_section'] = gr.Markdown(
                I18n.get('ext_correction_section', lang)
            )
                
            with gr.Row():
                components['checkbox_ext_wb'] = gr.Checkbox(
                    label=I18n.get('ext_wb', lang),
                    value=True
                )
                components['checkbox_ext_vignette'] = gr.Checkbox(
                    label=I18n.get('ext_vignette', lang),
                    value=False
                )
                
            components['slider_ext_zoom'] = gr.Slider(
                0.8, 1.2, 1.0, step=0.005,
                label=I18n.get('ext_zoom', lang)
            )
                
            components['slider_ext_distortion'] = gr.Slider(
                -0.2, 0.2, 0.0, step=0.01,
                label=I18n.get('ext_distortion', lang)
            )
                
            components['slider_ext_offset_x'] = gr.Slider(
                -30, 30, 0, step=1,
                label=I18n.get('ext_offset_x', lang)
            )
                
            components['slider_ext_offset_y'] = gr.Slider(
                -30, 30, 0, step=1,
                label=I18n.get('ext_offset_y', lang)
            )
                
            components['btn_ext_extract_btn'] = gr.Button(
                I18n.get('ext_extract_btn', lang),
                variant="primary",
                elem_classes=["primary-btn"]
            )
                
            components['textbox_ext_status'] = gr.Textbox(
                label=I18n.get('ext_status', lang),
                interactive=False
            )
            
        with gr.Column(scale=1):
            ext_hint = gr.Markdown(I18n.get('ext_hint_white', lang))
                
            ext_work_img = gr.Image(
                label=I18n.get('ext_marked', lang),
                show_label=False,
                interactive=True
            )
                
            with gr.Row():
                with gr.Column():
                    components['md_ext_sampling'] = gr.Markdown(
                        I18n.get('ext_sampling', lang)
                    )
                    ext_warp_view = gr.Image(show_label=False)
                    
                with gr.Column():
                    components['md_ext_reference'] = gr.Markdown(
                        I18n.get('ext_reference', lang)
                    )
                    ext_ref_view = gr.Image(
                        show_label=False,
                        value=ref_img,
                        interactive=False
                    )
                
            with gr.Row():
                with gr.Column():
                    components['md_ext_result'] = gr.Markdown(
                        I18n.get('ext_result', lang)
                    )
                    ext_lut_view = gr.Image(
                        show_label=False,
                        interactive=True
                    )
                    
                with gr.Column():
                    components['md_ext_manual_fix'] = gr.Markdown(
                        I18n.get('ext_manual_fix', lang)
                    )
                    ext_probe_html = gr.HTML(I18n.get('ext_click_cell', lang))
                        
                    ext_picker = gr.ColorPicker(
                        label=I18n.get('ext_override', lang),
                        value="#FF0000"
                    )
                        
                    components['btn_ext_apply_btn'] = gr.Button(
                        I18n.get('ext_apply_btn', lang)
                    )
                        
                    components['file_ext_download_npy'] = gr.File(
                        label=I18n.get('ext_download_npy', lang)
                    )
    
    # Event binding (simplified version, maintains original logic)
    ext_img_in.upload(
            on_extractor_upload,
            [ext_img_in, components['radio_ext_color_mode']],
            [ext_state_img, ext_work_img, ext_state_pts, ext_curr_coord, ext_hint]
    )
    
    components['radio_ext_color_mode'].change(
            on_extractor_mode_change,
            [ext_state_img, components['radio_ext_color_mode']],
            [ext_state_pts, ext_hint, ext_work_img]
    )
    
    components['btn_ext_rotate_btn'].click(
            on_extractor_rotate,
            [ext_state_img, components['radio_ext_color_mode']],
            [ext_state_img, ext_work_img, ext_state_pts, ext_hint]
    )
    
    ext_work_img.select(
            on_extractor_click,
            [ext_state_img, ext_state_pts, components['radio_ext_color_mode']],
            [ext_work_img, ext_state_pts, ext_hint]
    )
    
    components['btn_ext_reset_btn'].click(
            on_extractor_clear,
            [ext_state_img, components['radio_ext_color_mode']],
            [ext_work_img, ext_state_pts, ext_hint]
    )
    
    extract_inputs = [
            ext_state_img, ext_state_pts,
            components['slider_ext_offset_x'], components['slider_ext_offset_y'],
            components['slider_ext_zoom'], components['slider_ext_distortion'],
            components['checkbox_ext_wb'], components['checkbox_ext_vignette']
    ]
    extract_outputs = [
            ext_warp_view, ext_lut_view,
            components['file_ext_download_npy'], components['textbox_ext_status']
    ]
    
    components['btn_ext_extract_btn'].click(run_extraction, extract_inputs, extract_outputs)
    
    for s in [components['slider_ext_offset_x'], components['slider_ext_offset_y'],
                  components['slider_ext_zoom'], components['slider_ext_distortion']]:
            s.release(run_extraction, extract_inputs, extract_outputs)
    
    ext_lut_view.select(probe_lut_cell, [], [ext_probe_html, ext_picker, ext_curr_coord])
    components['btn_ext_apply_btn'].click(
            manual_fix_cell,
            [ext_curr_coord, ext_picker],
            [ext_lut_view, components['textbox_ext_status']]
    )
    
    return components



def create_about_tab_content(lang: str) -> dict:
    """Create about tab content"""
    components = {}
    
    # About page content (from i18n)
    components['md_about_content'] = gr.Markdown(I18n.get('about_content', lang))
    
    return components
