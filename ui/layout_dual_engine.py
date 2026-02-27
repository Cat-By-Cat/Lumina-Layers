# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    LUMINA STUDIO - DUAL ENGINE ARCHITECTURE                   ║
║                         双引擎架构 UI 布局文件                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Engine A: Lumina Classic (LUT 预计算模式)                                     ║
║  Engine B: Chroma K/S (物理动态模式)                                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Architecture:
- Top-level engine selector (Radio/Tabs)
- Conditional visibility for each engine's workflow
- Complete isolation between engines
- Shared styling and i18n support
"""

import gradio as gr
from core.i18n import I18n

# Import Lumina Classic components
from ui.layout_new import (
    create_calibration_tab_content,
    create_extractor_tab_content,
    create_converter_tab_content
)

# Import Chroma K/S components (to be created)
try:
    from core.ks_engine.ui_components import (
        create_ks_calibration_tab,
        create_ks_calculator_tab,
        create_ks_converter_tab,
        create_ks_filament_db_tab
    )
    HAS_KS_ENGINE = True
except ImportError:
    HAS_KS_ENGINE = False
    print("⚠️  Chroma K/S Engine not found. Only Lumina Classic will be available.")

from .styles import CUSTOM_CSS


def create_dual_engine_app(lang: str = "zh"):
    """
    创建双引擎应用主界面
    
    Args:
        lang: 语言设置 ("zh" 或 "en")
    
    Returns:
        gr.Blocks: Gradio 应用实例
    """
    
    with gr.Blocks(
        title="Lumina Studio - Dual Engine",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft()
    ) as app:
        
        # ========== 顶部标题 ==========
        gr.Markdown(
            """
            # 🌈 Lumina Studio - Dual Engine
            ### Multi-Material 3D Print Color System | 多材料3D打印色彩系统
            """
        )
        
        # ========== 全局引擎选择器 ==========
        with gr.Row():
            engine_selector = gr.Radio(
                choices=[
                    "🎨 Lumina Classic (LUT Mode)",
                    "⚗️ Chroma K/S (Physics Mode)"
                ],
                value="🎨 Lumina Classic (LUT Mode)",
                label="🔧 Core Processing Engine | 核心处理引擎",
                info="Select the color calculation method | 选择颜色计算方式",
                elem_classes=["engine-selector"]
            )
        
        gr.Markdown("---")
        
        # ========== Engine A: Lumina Classic ==========
        with gr.Group(visible=True) as lumina_classic_group:
            gr.Markdown(
                """
                ## 🎨 Lumina Classic Engine
                **LUT 预计算模式** - 基于物理校准色卡的快速映射系统
                
                - ✅ 无需复杂物理参数
                - ✅ 打印一次校准板即可使用
                - ✅ 支持 4色/6色/8色/黑白 多种模式
                - ✅ 适合快速迭代和批量生产
                """
            )
            
            with gr.Tabs():
                # Tab 1: 校准板生成器
                with gr.Tab("📐 Calibration Generator | 校准板生成"):
                    calibration_tab_components = create_calibration_tab_content(lang)
                
                # Tab 2: 颜色提取器
                with gr.Tab("🎨 Color Extractor | 颜色提取"):
                    extractor_tab_components = create_extractor_tab_content(lang)
                
                # Tab 3: 图像转换器
                with gr.Tab("🖼️ Image Converter | 图像转换"):
                    converter_tab_components = create_converter_tab_content(lang)
        
        # ========== Engine B: Chroma K/S ==========
        with gr.Group(visible=False) as chroma_ks_group:
            if HAS_KS_ENGINE:
                gr.Markdown(
                    """
                    ## ⚗️ Chroma K/S Engine
                    **物理动态模式** - 基于 Kubelka-Munk 光学理论的精确计算
                    
                    - 🔬 基于物理光学模型
                    - 🎯 更高的颜色准确度
                    - 🔄 支持任意耗材组合
                    - 📊 实时动态计算色域
                    """
                )
                
                with gr.Tabs():
                    # Tab 1: K/S 阶梯色卡生成
                    with gr.Tab("📏 K/S Step Card Generator | 阶梯色卡生成"):
                        ks_calibration_components = create_ks_calibration_tab()
                    
                    # Tab 2: K/S 拍照测算仪
                    with gr.Tab("📸 K/S Calculator | K/S 测算"):
                        ks_calculator_components = create_ks_calculator_tab()
                    
                    # Tab 3: 动态转换器
                    with gr.Tab("🎨 Dynamic Converter | 动态转换"):
                        ks_converter_components = create_ks_converter_tab()
                    
                    # Tab 4: 耗材数据库
                    with gr.Tab("💾 Filament Database | 耗材数据库"):
                        ks_filament_db_components = create_ks_filament_db_tab()
            else:
                gr.Markdown(
                    """
                    ## ⚠️ Chroma K/S Engine Not Available
                    
                    The K/S physics engine is not installed. To enable it:
                    
                    1. Ensure `core/ks_engine/` directory exists
                    2. Install required dependencies
                    3. Restart the application
                    
                    ---
                    
                    ## ⚠️ Chroma K/S 引擎未安装
                    
                    K/S 物理引擎尚未安装。要启用它：
                    
                    1. 确保 `core/ks_engine/` 目录存在
                    2. 安装所需依赖
                    3. 重启应用程序
                    """
                )
        
        # ========== 引擎切换逻辑 ==========
        def switch_engine(engine_choice):
            """
            切换引擎显示
            
            Args:
                engine_choice: 选择的引擎名称
            
            Returns:
                tuple: (lumina_visible, chroma_visible)
            """
            if "Lumina Classic" in engine_choice:
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        engine_selector.change(
            fn=switch_engine,
            inputs=[engine_selector],
            outputs=[lumina_classic_group, chroma_ks_group]
        )
        
        # ========== 底部信息 ==========
        gr.Markdown(
            """
            ---
            ### 📖 Documentation | 文档
            - [Lumina Classic Guide](https://github.com/MOVIBALE/Lumina-Layers)
            - [Chroma K/S Theory](https://github.com/borealis-zhe/ChromaStack)
            
            ### 📜 License | 许可证
            - Lumina Classic: CC BY-NC-SA 4.0
            - Chroma K/S: GPL-3.0
            """
        )
    
    return app


if __name__ == "__main__":
    app = create_dual_engine_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
