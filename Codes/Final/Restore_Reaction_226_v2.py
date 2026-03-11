"""
等离子体刻蚀仿真 - 改进版 v2
改进内容：
1. 添加物理常量定义模块，消除魔法数字
2. 支持五槽结构（160/140/120/100/80 nm）
3. 添加离子能量追踪
4. 改进反射概率模型（基于文献）
5. 改进刻蚀产额模型（Y ∝ √E）
6. 增强 Bowing 效应
直接按照SEM电镜的轮廓进行仿真，但是没有什么鸟用。

参考文献：
- Modeling of microtrenching and bowing effects in nanoscale Si ICP etching
- Profile simulation of high aspect ratio contact etch
- Effect of Mask Geometry Variation on Plasma Etching Profiles
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
import math
import time
import os
import csv

# ================= 1. 环境初始化 =================

# Taichi GPU 初始化，带有重试机制
try:
    ti.init(arch=ti.gpu, device_memory_fraction=0.8)
    print("[OK] GPU 初始化成功")
except KeyboardInterrupt:
    print("[ERROR] GPU 初始化被中断，切换到 CPU 模式")
    ti.init(arch=ti.cpu)
except Exception as e:
    print(f"[ERROR] GPU 初始化失败: {e}，切换到 CPU 模式")
    ti.init(arch=ti.cpu)

# --- 路径配置 ---
SAVE_DIR = r"Csv\Test_MarchingSquares_2026"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ================= 物理常量配置 =================

# --- 材质类型 ---
MAT_VACUUM = 0
MAT_SI = 1      # 硅衬底
MAT_MASK = 2    # 掩膜

# --- 几何常量（根据 SEM 图像调整） ---
GRID_SCALE_NM = 0.2  # 1 网格 = 0.2nm
TRENCH_WIDTHS = [160, 140, 120, 100, 80]  # 五个槽的宽度
PILLAR_WIDTH = 100    # 中间支柱宽度（nm）
SIDE_MASK_WIDTH = 400 # 两侧掩膜宽度（nm）
MASK_THICKNESS = 130  # 掩膜厚度（nm，网格单位）
TAPER_ANGLE_DEG = 5  # 掩膜倾角（度）

# --- 网格尺寸 ---
# 根据 5 槽结构计算所需网格宽度
total_width_nm = (SIDE_MASK_WIDTH * 2 +
                  sum(TRENCH_WIDTHS) +
                  PILLAR_WIDTH * (len(TRENCH_WIDTHS) - 1))
ROWS = int(total_width_nm / GRID_SCALE_NM) + 50  # 预留余量
COLS = 700  # 保持原有高度

# --- 真空区域 ---
vacuum = 100  # 网格单位
deep_border = 230  # 掩膜底边界

# --- 粒子属性 ---
ION_MASS_RATIO = 35.45 / 28.09  # Cl/Si 质量比
ENERGY_INITIAL = 1.0            # 初始归一化能量（100%）
ENERGY_MIN = 0.05               # 最小能量阈值（低于此值不再刻蚀）

# --- 角度分布参数（高斯分布） ---
ION_ANGLE_SIGMA = 1.91 * (math.pi / 180)   # 离子入射角标准差（弧度）
NEUTRAL_ANGLE_SIGMA = 7.64 * (math.pi / 180)  # 中性粒子入射角标准差（弧度）

# --- 反射概率参数 ---
REFLECTION_THRESHOLD_ANGLE = math.pi / 3  # 60°
# 离子反射概率：掠角易反射，cos(theta) < threshold 时线性增长
# 掩膜反射：保底概率更高（硬质材料）

# --- 能量损失系数 ---
ENERGY_LOSS_MASK_REFLECTION = 0.90   # 掩膜反射保留 90% 能量
ENERGY_LOSS_SI_REFLECTION = 0.40     # Si 侧壁反射保留 40% 能量（增强 bowing）

# --- 刻蚀产额参数 ---
ETCH_YIELD_BASE = 0.1
ETCH_YIELD_CL_ENHANCEMENT = 0.2
ETCH_YIELD_MASK_SCALING = 0.2  # 掩膜刻蚀产额缩放
ETCH_YIELD_REFLECTED_SCALING = 0.7  # 反射后刻蚀产额（保留部分能量用于 bowing）

# --- 中性粒子吸附 ---
CL_SATURATION = 3      # Cl 饱和阈值
ADSORPTION_PROB = 0.7  # 中性粒子吸附概率

# --- 仿真参数 ---
TOTAL_PARTICLES = 100000000
BATCH_SIZE = 4000
RATIO = 10.0 / 11.0  # 中性/总粒子比例（10/11 ≈ 91% 中性）
SYNC_INTERVAL = 50  # 每 N 个批次同步一次

# --- Taichi 数据场 ---
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))
grid_temp = ti.field(dtype=ti.f32, shape=(ROWS, COLS))


# ================= 2. 物理辅助函数 =================

@ti.func
def calculate_reflection_probability(mat: int, theta_coll: float, is_ion: int, cl_coverage: int):
    """
    计算反射概率

    Args:
        mat: 材质类型 (MAT_SI=1, MAT_MASK=2)
        theta_coll: 碰撞角（弧度，0=垂直，pi/2=掠角）
        is_ion: 是否为离子 (1=离子, 0=中性)
        cl_coverage: Cl 覆盖度 (0-4)

    Returns:
        反射概率 (0-1)
    """
    prob_reflect = 0.0

    if is_ion == 1:
        # 离子反射模型：掠角易反射
        # theta_coll 从 REFLECTION_THRESHOLD_ANGLE (60°) 到 pi/2 (90°)
        # 线性增长到 1.0
        threshold = REFLECTION_THRESHOLD_ANGLE
        angle_range = math.pi / 2 - threshold

        if angle_range > 1e-6:
            angle_factor = ti.max(0.0, (theta_coll - threshold) / angle_range)
            angle_factor = ti.min(1.0, angle_factor)
            prob_reflect = angle_factor

        # 掩膜反射概率更高（硬质材料）
        if mat == MAT_MASK:
            # 保底概率 0.8，额外增加 0.2
            prob_reflect = ti.min(1.0, prob_reflect + 0.4)
            prob_reflect = ti.max(0.8, prob_reflect)

    else:  # 中性粒子
        # Lambertian 漫反射，粘附系数与 Cl 覆盖度相关
        # Cl 越多，越难粘附，反射概率越高
        prob_reflect = ADSORPTION_PROB + 0.1 * cl_coverage
        prob_reflect = ti.min(0.95, prob_reflect)

    return prob_reflect


@ti.func
def calculate_etch_yield(mat: int, cl_coverage: int, energy: float, incident_angle: float, ref_count: int):
    """
    计算刻蚀产额

    Args:
        mat: 材质类型
        cl_coverage: Cl 覆盖度
        energy: 离子能量（归一化 0-1）
        incident_angle: 入射角（弧度）
        ref_count: 反射次数

    Returns:
        刻蚀产额 (0-1)
    """
    # 基础产额
    prob_etch = ETCH_YIELD_BASE

    # Cl 增强效应（化学刻蚀）
    if cl_coverage > 0:
        prob_etch += ETCH_YIELD_CL_ENHANCEMENT

    # 能量依赖：Y ∝ √E（物理文献支持）
    prob_etch *= ti.sqrt(energy)

    # 掩膜刻蚀产额更低（硬质材料）
    if mat == MAT_MASK:
        prob_etch *= ETCH_YIELD_MASK_SCALING

    # 反射后的刻蚀产额调整（保留部分能量用于 bowing）
    if ref_count >= 1:
        prob_etch *= ETCH_YIELD_REFLECTED_SCALING  # 0.7

    # 角度依赖：掠角刻蚀效率低
    # 当入射角 > 40° 时线性下降
    angle_deg = incident_angle * 180.0 / math.pi
    angle_threshold = 40.0
    if angle_deg > angle_threshold:
        angle_penalty = 1.0 - (angle_deg - angle_threshold) / 50.0  # 40°时1.0，90°时0.0
        angle_penalty = ti.max(0.2, angle_penalty)
        prob_etch *= angle_penalty

    return ti.max(0.0, prob_etch)


@ti.func
def update_ion_energy(current_energy: float, mat_of_surface: int):
    """
    更新离子能量（反射后）

    Args:
        current_energy: 当前能量
        mat_of_surface: 碰撞表面材质

    Returns:
        更新后的能量
    """
    new_energy = current_energy
    if mat_of_surface == MAT_MASK:
        # 掩膜反射：保留 90% 能量
        new_energy = current_energy * ENERGY_LOSS_MASK_REFLECTION
    elif mat_of_surface == MAT_SI:
        # Si 侧壁反射：保留 40% 能量（用于 bowing）
        new_energy = current_energy * ENERGY_LOSS_SI_REFLECTION

    # 确保不低于最小阈值
    new_energy = ti.max(ENERGY_MIN, new_energy)
    return new_energy


@ti.func
def get_surface_normal(px: int, py: int):
    """ 计算表面法线 (指向真空) """
    nx, ny = 0.0, 0.0
    for i, j in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        if 0 <= px + i < ROWS and 0 <= py + j < COLS:
            if grid_exist[px + i, py + j] < 0.5:
                nx += float(i)
                ny += float(j)
    norm = ti.sqrt(nx**2 + ny**2) + 1e-6
    return nx / norm, ny / norm


@ti.func
def get_reflection_vector(vx: float, vy: float, nx: float, ny: float, is_ion: int):
    """ 计算反射向量 """
    rvx, rvy = 0.0, 0.0
    if is_ion == 1:
        # 镜面反射 (v - 2(v.n)n)
        dot = vx * nx + vy * ny
        rvx = vx - 2 * dot * nx
        rvy = vy - 2 * dot * ny
    else:
        # Lambertian 漫反射
        tx, ty = -ny, nx
        sin_theta = (ti.random() - 0.5) * 2.0
        cos_theta = ti.sqrt(1.0 - sin_theta**2)
        rvx = nx * cos_theta + tx * sin_theta
        rvy = ny * cos_theta + ty * sin_theta
    return rvx, rvy


@ti.kernel
def smooth_grid():
    """ 表面平滑 (防止数值噪声) """
    w_center = 0.98
    w_neighbor = (1.0 - w_center) / 4.0
    for i, j in grid_exist:
        if 1 <= i < ROWS - 1 and 1 <= j < COLS - 1:
            val = (grid_exist[i, j] * w_center +
                   (grid_exist[i+1, j] + grid_exist[i-1, j] +
                    grid_exist[i, j+1] + grid_exist[i, j-1]) * w_neighbor)
            grid_temp[i, j] = val
        else:
            grid_temp[i, j] = grid_exist[i, j]
    for i, j in grid_exist:
        grid_exist[i, j] = grid_temp[i, j]


# ================= 3. 几何初始化（支持五槽） =================

def init_grid():
    """
    初始化五槽结构
    槽宽度：160/140/120/100/80 nm
    支柱宽度：100 nm
    两侧掩膜：400 nm
    掩膜倾角：15°
    """
    angle_rad = TAPER_ANGLE_DEG * math.pi / 180.0
    k_mask = math.tan(angle_rad)
    
    # 使用NumPy数组以加速初始化
    exist_array = np.zeros((ROWS, COLS), dtype=np.float32)
    material_array = np.zeros((ROWS, COLS), dtype=np.int32)
    count_cl_array = np.zeros((ROWS, COLS), dtype=np.int32)
    
    # 预处理：计算所有槽和支柱的边界
    trench_regions = []  # [(left, right, pillar_width), ...]
    current_x = int(SIDE_MASK_WIDTH / GRID_SCALE_NM)
    
    for n in range(len(TRENCH_WIDTHS)):
        trench_width = TRENCH_WIDTHS[n]
        trench_width_grid = int(trench_width / GRID_SCALE_NM)
        pillar_width_grid = int(PILLAR_WIDTH / GRID_SCALE_NM)
        
        trench_left = current_x
        trench_right = current_x + trench_width_grid
        
        trench_regions.append((trench_left, trench_right, pillar_width_grid))
        current_x = trench_right + pillar_width_grid
    
    rightmost_x = current_x
    left_mask_boundary = int(SIDE_MASK_WIDTH / GRID_SCALE_NM)
    
    # 遍历所有网格点
    for j in range(COLS):
        for i in range(ROWS):
            # 真空区域
            if j <= vacuum:
                exist_array[i, j] = 0.0
                material_array[i, j] = MAT_VACUUM
                continue
            
            # 初始化为真空
            is_filled = False
            material = MAT_VACUUM
            
            if j < deep_border:
                # 左侧掩膜
                if i < left_mask_boundary:
                    is_filled = True
                    material = MAT_MASK
                # 右侧掩膜
                elif i >= rightmost_x:
                    is_filled = True
                    material = MAT_MASK
                else:
                    # 检查是否在槽中
                    for trench_left, trench_right, pillar_width_grid in trench_regions:
                        pillar_half = int(pillar_width_grid / 2)
                        
                        # 掩膜侧壁（带倾角）
                        offset = int((deep_border - j) * k_mask)
                        l_cur = max(0, min(trench_left - offset, ROWS - 1))
                        r_cur = max(0, min(trench_right + offset, ROWS - 1))
                        
                        # 支柱范围
                        l_side = max(0, min(trench_left - pillar_half, ROWS - 1))
                        r_side = max(0, min(trench_right + pillar_half, ROWS - 1))
                        
                        # 检查i是否在支柱范围内
                        if l_side <= i < r_side:
                            # 如果在槽内部（l_cur < i < r_cur），则为真空
                            if l_cur < i < r_cur:
                                is_filled = False
                                material = MAT_VACUUM
                            # 否则为掩膜
                            else:
                                is_filled = True
                                material = MAT_MASK
                            break
            
            # Si 衬底
            if j >= deep_border and is_filled == False:
                is_filled = True
                material = MAT_SI
            
            exist_array[i, j] = 1.0 if is_filled else 0.0
            material_array[i, j] = material
    
    # 将NumPy数组复制到Taichi字段
    grid_exist.from_numpy(exist_array)
    grid_material.from_numpy(material_array)
    grid_count_cl.from_numpy(count_cl_array)


# ================= 4. 核心仿真逻辑（反射优先 + 能量追踪） =================

@ti.kernel
def simulate_batch():
    """
    模拟一批粒子的运动和反应
    关键改进：
    1. 添加离子能量追踪
    2. 改进反射概率模型
    3. 改进刻蚀产额模型（Y ∝ √E）
    4. 增强 bowing 效应
    """
    for k in range(BATCH_SIZE):
        # --- A. 粒子生成 ---
        px, py = ti.random() * (ROWS - 1), 1.0

        is_ion = 0
        if ti.random() > RATIO:
            is_ion = 1

        # 角度分布（高斯）
        sigma = ION_ANGLE_SIGMA if is_ion == 1 else NEUTRAL_ANGLE_SIGMA
        angle = ti.randn() * sigma
        angle = ti.max(-math.pi / 2, ti.min(math.pi / 2, angle))
        vx, vy = ti.sin(angle), ti.cos(angle)

        # 离子能量追踪
        energy = ENERGY_INITIAL

        alive = True
        steps = 0
        ref_count = 0  # 反射计数

        while alive and steps < 3000:
            steps += 1
            # 移动
            px_n, py_n = px + vx * 1.1, py + vy * 1.1

            # 周期性边界（x 方向）
            if px_n < 0:
                px_n += ROWS
            if px_n >= ROWS:
                px_n -= ROWS

            # y 方向越界检查
            if py_n < 0 or py_n >= COLS:
                alive = False
                break

            ipx, ipy = int(px_n), int(py_n)

            # --- B. 碰撞检测 ---
            if grid_exist[ipx, ipy] > 0.5:
                mat = grid_material[ipx, ipy]
                cl_n = grid_count_cl[ipx, ipy]

                # 计算表面法线和碰撞角
                nx, ny = get_surface_normal(ipx, ipy)
                cos_theta = -(vx * nx + vy * ny)
                if cos_theta < 0:
                    cos_theta = 0.0
                if cos_theta > 1:
                    cos_theta = 1.0

                theta_coll = ti.acos(cos_theta)

                # === 判定顺序：反射 -> 反应 ===

                did_reflect = False

                # --- 1. 反射判定（改进的概率模型） ---
                prob_reflect = calculate_reflection_probability(mat, theta_coll, is_ion, cl_n)

                if ti.random() < prob_reflect:
                    did_reflect = True
                    ref_count += 1

                    # 离子反射后更新能量
                    if is_ion == 1:
                        energy = update_ion_energy(energy, mat)

                # --- 2. 行为分支 ---
                if did_reflect:
                    # >> 反射物理 <<
                    vx, vy = get_reflection_vector(vx, vy, nx, ny, is_ion)
                    px = px + nx * 1.5
                    py = py + ny * 1.5
                else:
                    # >> 反应判定（被捕获） <<

                    if is_ion == 1:
                        # 离子刻蚀（改进的产额模型）
                        prob_etch = calculate_etch_yield(mat, cl_n, energy, theta_coll, ref_count)

                        if ti.random() < prob_etch:
                            grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - 0.2)
                            if grid_exist[ipx, ipy] <= 0.0:
                                grid_material[ipx, ipy] = MAT_VACUUM
                            alive = False
                        else:
                            alive = False  # 沉积/消失

                    else:
                        # 中性粒子反应
                        prob_etch = 0.0

                        # 只有 Cl 饱和才刻蚀
                        if mat == MAT_SI and cl_n >= CL_SATURATION:
                            prob_etch = 0.1 * 0.1

                        if ti.random() < prob_etch:
                            # 刻蚀
                            grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - 0.2)
                            if grid_exist[ipx, ipy] <= 0.0:
                                grid_material[ipx, ipy] = MAT_VACUUM
                                grid_count_cl[ipx, ipy] = 0
                            alive = False
                        else:
                            # 吸附
                            prob_adsorb = 1.0 - cl_n / 4.0
                            if cl_n < CL_SATURATION and ti.random() < prob_adsorb * 0.1:
                                grid_count_cl[ipx, ipy] += 1
                            alive = False
            else:
                px, py = px_n, py_n


# ================= 5. 轮廓提取（元胞法） =================

def get_contour_points(raw_grid):
    """
    使用 Marching Squares (元胞法) 提取等值面轮廓
    """
    # 先做轻微高斯平滑，避免网格噪声导致轮廓破碎
    smoothed = gaussian_filter(raw_grid.astype(float), sigma=1.0)

    contour_points = []

    if HAS_SKIMAGE:
        # 方法 A: 使用 skimage.measure.find_contours (标准元胞法)
        contours = measure.find_contours(smoothed, 0.5)

        for contour in contours:
            # 交换列(x)和行(y) -> (x, y)
            for p in contour:
                contour_points.append((p[1], p[0]))

    else:
        # 方法 B: 备用方案 (Matplotlib Contour Engine)
        fig_temp = plt.figure()
        ax_temp = fig_temp.add_subplot(111)
        cnt = ax_temp.contour(smoothed.T, levels=[0.5])

        for path in cnt.collections[0].get_paths():
            verts = path.vertices
            for v in verts:
                contour_points.append((v[0], v[1]))
        plt.close(fig_temp)

    return contour_points


def save_csv(points, filename):
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(points)


# ================= 6. 工具函数 =================

def print_simulation_parameters():
    """ 打印仿真参数配置 """
    print("=" * 70)
    print("  等离子体刻蚀仿真 - 改进版 v2")
    print("=" * 70)
    print(f"网格尺寸: {ROWS} × {COLS}")
    print(f"网格比例: {GRID_SCALE_NM} nm/pixel")
    print(f"槽数量: {len(TRENCH_WIDTHS)}")
    print(f"槽宽度: {TRENCH_WIDTHS} nm")
    print(f"支柱宽度: {PILLAR_WIDTH} nm")
    print(f"两侧掩膜: {SIDE_MASK_WIDTH} nm")
    print(f"掩膜厚度: {MASK_THICKNESS} px")
    print(f"掩膜倾角: {TAPER_ANGLE_DEG}°")
    print("-" * 70)
    print(f"粒子总数: {TOTAL_PARTICLES:,}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"离子比例: {(1.0 - RATIO):.1%}")
    print(f"中性比例: {RATIO:.1%}")
    print("-" * 70)
    print(f"能量损失 - 掩膜: {ENERGY_LOSS_MASK_REFLECTION}")
    print(f"能量损失 - Si: {ENERGY_LOSS_SI_REFLECTION}")
    print(f"反射后刻蚀产额: {ETCH_YIELD_REFLECTED_SCALING}")
    print(f"Cl 饱和阈值: {CL_SATURATION}")
    print("=" * 70)


# ================= 7. 主程序 =================

def main():
    if not HAS_SKIMAGE:
        print("Warning: 未检测到 scikit-image，将使用 Matplotlib 引擎提取轮廓。")
        print("建议安装: pip install scikit-image")
        print()

    # 打印仿真参数
    print_simulation_parameters()
    print()

    # 初始化几何结构
    print("正在初始化几何结构...")
    init_grid()
    print("几何结构初始化完成。")
    print()

    # 设置交互式绘图
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    history_lines = []

    print(">>> 模拟开始（能量追踪 + 改进物理模型） <<<")
    print()

    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    start_time = time.time()

    for i in range(num_batches):
        simulate_batch()
        smooth_grid()

        if i % SYNC_INTERVAL == 0:
            ti.sync()

            mat_data = grid_material.to_numpy()
            exist_data = grid_exist.to_numpy()

            # 使用元胞法提取轮廓
            points = get_contour_points(exist_data)

            current_count = i * BATCH_SIZE
            save_csv(points, f"contour_{current_count}.csv")

            # 提取 x, y 用于绘图
            if len(points) > 0:
                lx = [p[0] for p in points]
                ly = [p[1] for p in points]
                history_lines.append((lx, ly))

            # 绘图
            ax.clear()

            # 创建 RGB 图像
            rgb = np.zeros((ROWS, COLS, 3), dtype=np.float32)
            vac_mask = (exist_data < 0.5)
            mask_mask = (exist_data >= 0.5) & (mat_data == MAT_MASK)
            si_mask = (exist_data >= 0.5) & (mat_data == MAT_SI)

            rgb[vac_mask] = to_rgb("#008CFF")   # 真空：蓝色
            rgb[mask_mask] = to_rgb("#00FFFF")  # 掩膜：青色
            rgb[si_mask] = to_rgb("#00008B")    # Si：深蓝色

            ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='upper')

            # 绘制历史轮廓
            for idx, (hx, hy) in enumerate(history_lines):
                alpha = 0.3 + 0.7 * (idx / len(history_lines))
                color = 'white' if idx == len(history_lines) - 1 else 'red'
                ax.plot(hy, hx, color=color, alpha=alpha, linewidth=1.0, linestyle='-')

            # 计算进度和时间
            progress = i / num_batches
            elapsed = time.time() - start_time
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0

            ax.set_title(
                f"Simulation v2: {current_count}/{TOTAL_PARTICLES} ({progress:.1%})\n"
                f"Method: Energy Tracking + Improved Physics\n"
                f"Bowing Enhanced | 5 Trenches: {TRENCH_WIDTHS} nm"
            )
            ax.set_xlim(0, ROWS)
            ax.set_ylim(COLS, 0)

            plt.pause(0.01)

            # 打印进度
            print(f"进度: {progress:.1%} | 已用: {elapsed:.0f}s | 预计剩余: {eta:.0f}s", end='\r')

    ti.sync()

    # 保存最终轮廓
    final_points = get_contour_points(grid_exist.to_numpy())
    save_csv(final_points, "contour_final.csv")

    elapsed = time.time() - start_time
    print(f"\n>>> 模拟完成。总用时: {elapsed:.0f}s ({elapsed/60:.1f} 分钟)")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
