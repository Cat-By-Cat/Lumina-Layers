import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
#               全局配置参数
# ==========================================

# --- 输入/输出设置 ---
IMAGE_PATH = r'C:\Users\MIN\Downloads\IMG_0545.JPG'  # <--- 请在此处修改你的图片路径

# --- 3D打印/物理参数 ---
LAYER_HEIGHT = 0.08                    # 打印层高 (mm)
NUM_STEPS = 5                         # 色卡台阶数 (1层到5层)
BACKING_REFLECTANCE_WHITE = 0.94      # A4纸作为底材的反射率 (0.92-0.96)
BACKING_REFLECTANCE_BLACK = 0.00      # 黑底反射率 (假设完全吸光)

# --- 图像处理参数 ---
A4_WIDTH = 1414                       # A4纸透视变换后的宽度 (像素)
A4_HEIGHT = 1000                      # A4纸透视变换后的高度
CHIP_W, CHIP_H = 400, 500             # 色卡透视变换后的尺寸

# ==========================================
#           第一部分：图像处理工具
# ==========================================

def interactive_select_corners(img, window_name="Select Corners"):
    """
    交互式选取四个角点
    """
    h, w = img.shape[:2]
    # 如果图片太大，缩放显示以便操作
    scale = 800 / h if h > 800 else 1.0
    display_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    temp_img = display_img.copy()
    
    print(f"\n🖱️  [{window_name}] 请依次点击: 左上 -> 右上 -> 右下 -> 左下")
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x/scale), int(y/scale)])
            cv2.circle(temp_img, (x, y), 5, (0, 0, 255), -1)
            # 画个序号
            cv2.putText(temp_img, str(len(points)), (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow(window_name, temp_img)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, temp_img)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # 等待直到选够4个点并按任意键，或直接按ESC退出
    while True:
        k = cv2.waitKey(100)
        if len(points) == 4:
            # 简单反馈一下已选完
            cv2.putText(temp_img, "Done! Press Any Key", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, temp_img)
            cv2.waitKey(0)
            break
        if k == 27: # ESC
            cv2.destroyAllWindows()
            return None
            
    cv2.destroyAllWindows()
    return np.float32(points)

def apply_perspective_transform(img, src_pts, dst_w, dst_h):
    """
    透视变换矫正
    """
    dst_pts = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (dst_w, dst_h))

def auto_white_balance_by_paper(img_a4):
    """
    基于A4纸边缘区域进行白平衡
    """
    h, w = img_a4.shape[:2]
    margin_h, margin_w = int(h * 0.1), int(w * 0.1)
    
    # 创建掩膜，只取边缘部分（认为是纯白纸区域，避开中间的色卡）
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (w, margin_h), 255, -1)
    cv2.rectangle(mask, (0, h-margin_h), (w, h), 255, -1)
    cv2.rectangle(mask, (0, 0), (margin_w, h), 255, -1)
    cv2.rectangle(mask, (w-margin_w, 0), (w, h), 255, -1)
    
    mean_bg_bgr = cv2.mean(img_a4, mask=mask)[:3]
    print(f"📄 A4纸参考色 (RGB): {np.round(mean_bg_bgr[::-1]).astype(int)}")
    
    # 计算增益 (目标是 RGB 都达到 250，留一点余量防止过曝)
    gains = 250.0 / (np.array(mean_bg_bgr) + 1e-5)
    
    # 应用增益
    return np.clip(cv2.multiply(img_a4.astype(float), gains), 0, 255).astype(np.uint8)

def process_image_to_data(image_path):
    """
    核心图像处理流程：读取 -> 校正 -> 采样 -> 返回DataFrame
    """
    if not os.path.exists(image_path):
        print(f"❌ 找不到图片: {image_path}"); return None

    raw_img = cv2.imread(image_path)
    
    # 1. A4 校正
    print("\n--- [Step 1] A4 纸校正 ---")
    pts_a4 = interactive_select_corners(raw_img, "1. Click A4 Paper Corners")
    if pts_a4 is None: return None
    
    img_a4 = apply_perspective_transform(raw_img, pts_a4, A4_WIDTH, A4_HEIGHT)
    img_calibrated = auto_white_balance_by_paper(img_a4)
    # cv2.imwrite("debug_step1_a4_balanced.jpg", img_calibrated) # 可选：保存调试图
    
    # 2. 样片提取
    print("\n--- [Step 2] 样片提取 ---")
    print("⚠️  请点击样片四周：确保上面是厚端(5层)，下面是薄端(1层)")
    pts_chip = interactive_select_corners(img_calibrated, "2. Click Chip Corners (Top=Thick, Bottom=Thin)")
    if pts_chip is None: return None
    
    img_chip = apply_perspective_transform(img_calibrated, pts_chip, CHIP_W, CHIP_H)
    # cv2.imwrite("debug_step2_chip_flat.jpg", img_chip) # 可选：保存调试图
    
    # 3. 采样数据
    rows = NUM_STEPS
    cols = 2
    dy = CHIP_H // rows
    dx = CHIP_W // cols
    
    data = []
    debug_view = img_chip.copy()
    
    print("\n🔍 开始采样 (逻辑: 图像从上到下 row0->row4, 对应层数 5->1)...")
    
    for r in range(rows):
        # 几何计算
        x_left = int(0.5 * dx)   # 黑底中心
        x_right = int(1.5 * dx)  # 白底中心
        y_center = int((r + 0.5) * dy)
        
        patch_size = 20
        
        # 提取颜色区域
        roi_0 = img_chip[y_center-patch_size:y_center+patch_size, x_left-patch_size:x_left+patch_size]
        rgb_0 = np.mean(roi_0, axis=(0,1))[::-1] # BGR转RGB
        
        roi_w = img_chip[y_center-patch_size:y_center+patch_size, x_right-patch_size:x_right+patch_size]
        rgb_w = np.mean(roi_w, axis=(0,1))[::-1]
        
        # Linear Reflectance (反伽马校正)
        R0_linear = (rgb_0 / 255.0) ** 2.2
        Rw_linear = (rgb_w / 255.0) ** 2.2
        
        # === 核心逻辑映射 ===
        # r=0 (图片最上方) -> 实物第 5 层 (最厚)
        # r=4 (图片最下方) -> 实物第 1 层 (最薄)
        layer_idx = NUM_STEPS - r 
        
        print(f"  - 扫描行 {r}: 对应实际层数 {layer_idx}")
        
        data.append({
            'Layer_Index': layer_idx,
            'R0_r': R0_linear[0], 'R0_g': R0_linear[1], 'R0_b': R0_linear[2],
            'Rw_r': Rw_linear[0], 'Rw_g': Rw_linear[1], 'Rw_b': Rw_linear[2]
        })
        
        # 绘制调试圆点
        cv2.circle(debug_view, (x_left, y_center), 5, (0,255,0), -1)
        cv2.circle(debug_view, (x_right, y_center), 5, (0,0,255), -1)

    os.makedirs("debug_output", exist_ok=True)
    cv2.imwrite("debug_output/debug_step3_sampling.jpg", debug_view)

    # 排序并生成DataFrame
    df = pd.DataFrame(data).sort_values('Layer_Index')
    
    return df

# ==========================================
#           第二部分：K-M 理论拟合
# ==========================================

def km_reflectance(K, S, h, Rg):
    """
    Kubelka-Munk 理论反射率公式
    """
    S = max(S, 1e-6) # 避免除零
    
    a = 1 + (K / S)
    b = np.sqrt(a**2 - 1)
    
    bSh = b * S * h
    sinh_bSh = np.sinh(bSh)
    cosh_bSh = np.cosh(bSh)
    
    numerator = sinh_bSh * (1 - Rg * a) + Rg * b * cosh_bSh
    denominator = sinh_bSh * (a - Rg) + b * cosh_bSh
    
    R = numerator / denominator
    return R

def fit_km_parameters(thicknesses, R0_measured, Rw_measured):
    """
    针对单个颜色通道拟合 K 和 S
    """
    # 初始猜测 [K, S]
    x0 = [0.1, 1.0] 
    
    def loss_function(params):
        K_val, S_val = params
        
        # 预测黑底 (Rg=0) 和 白底 (Rg=White)
        R0_pred = km_reflectance(K_val, S_val, thicknesses, BACKING_REFLECTANCE_BLACK)
        Rw_pred = km_reflectance(K_val, S_val, thicknesses, BACKING_REFLECTANCE_WHITE)
        
        # MSE 误差
        error_0 = np.mean((R0_pred - R0_measured) ** 2)
        error_w = np.mean((Rw_pred - Rw_measured) ** 2)
        return error_0 + error_w

    # 约束: K, S 必须 > 0
    bounds = [(1e-5, 100), (1e-5, 100)]
    
    result = minimize(loss_function, x0, bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun

def calculate_and_plot_km(df):
    """
    主计算流程
    """
    print("\n" + "="*50)
    print("🚀 开始 Kubelka-Munk 参数拟合...")
    
    # 准备数据
    thicknesses = df['Layer_Index'].values * LAYER_HEIGHT
    print(f"   厚度范围: {thicknesses[0]:.1f}mm - {thicknesses[-1]:.1f}mm")
    
    results = {}
    channels = ['r', 'g', 'b']
    
    # 创建可视化图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, ch in enumerate(channels):
        print(f"\n🎨 正在处理 {ch.upper()} 通道...")
        
        R0_meas = df[f'R0_{ch}'].values
        Rw_meas = df[f'Rw_{ch}'].values
        
        # 拟合
        (best_K, best_S), error = fit_km_parameters(thicknesses, R0_meas, Rw_meas)
        results[ch] = {'K': best_K, 'S': best_S}
        
        print(f"   ✅ K={best_K:.4f}, S={best_S:.4f} (Error: {error:.5f})")
        
        # --- 绘图 ---
        ax = axes[i]
        # 散点：测量值
        ax.scatter(thicknesses, R0_meas, color='black', label='Meas (Black Base)')
        ax.scatter(thicknesses, Rw_meas, color='gray', marker='s', label='Meas (White Base)')
        
        # 曲线：拟合模型
        h_smooth = np.linspace(0, thicknesses[-1] + 0.2, 50)
        R0_smooth = km_reflectance(best_K, best_S, h_smooth, BACKING_REFLECTANCE_BLACK)
        Rw_smooth = km_reflectance(best_K, best_S, h_smooth, BACKING_REFLECTANCE_WHITE)
        
        plot_color = 'red' if ch=='r' else 'green' if ch=='g' else 'blue'
        ax.plot(h_smooth, R0_smooth, linestyle='--', color=plot_color, label='K-M Model (Black)')
        ax.plot(h_smooth, Rw_smooth, linestyle='-', color=plot_color, alpha=0.5, label='K-M Model (White)')
        
        ax.set_title(f"Channel {ch.upper()}\nK={best_K:.2f}, S={best_S:.2f}")
        ax.set_xlabel("Thickness (mm)")
        ax.set_ylabel("Reflectance")
        if i == 0: ax.legend()

    plt.tight_layout()
    os.makedirs("debug_output", exist_ok=True)
    plt.savefig("debug_output/km_fitting_result.png")
    print("\n" + "="*50)
    print(f"📈 拟合曲线图已保存至: debug_output/km_fitting_result.png")

    # 输出 JSON
    print("-" * 50)
    print("📋 最终 JSON 参数 (可直接填入 filaments.json):")
    print("{")
    print(f'  "FILAMENT_K": [{results["r"]["K"]:.4f}, {results["g"]["K"]:.4f}, {results["b"]["K"]:.4f}],')
    print(f'  "FILAMENT_S": [{results["r"]["S"]:.4f}, {results["g"]["S"]:.4f}, {results["b"]["S"]:.4f}]')
    print("}")
    
    # 物理意义解读
    avg_S = np.mean([results[c]['S'] for c in channels])
    avg_K = np.mean([results[c]['K'] for c in channels])
    
    print("-" * 50)
    print("💡 材料特性解读:")
    if avg_S > 10: print("   [高遮盖力] 类似牛奶或浓缩颜料，薄层即可遮盖底色。")
    elif avg_S < 1: print("   [低遮盖力] 类似清漆或彩色玻璃，需要很厚才能遮盖底色。")
    else: print("   [半透明] 类似玉石或雾状塑料。")
    
    if avg_K > 2: print("   [深色] 吸光能力强。")
    elif avg_K < 0.1: print("   [浅色/透明] 吸光能力弱。")
    print("="*50)

# ==========================================
#               主程序入口
# ==========================================

def main():
    print("=== 3D打印耗材 K-M 参数校准全流程 ===")
    
    # 1. 处理图片提取数据
    df = process_image_to_data(IMAGE_PATH)
    
    if df is None:
        print("❌ 图片处理失败或已取消，程序终止。")
        return

    # 2. 计算 K-M 参数
    calculate_and_plot_km(df)

if __name__ == "__main__":
    main()