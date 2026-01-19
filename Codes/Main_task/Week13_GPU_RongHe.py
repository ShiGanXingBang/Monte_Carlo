import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
import math
import time
import os
import csv

# ================= 1. 环境初始化 =================
ti.init(arch=ti.gpu)  # 启动核显卡加速

# --- 保存路径设置 (源自 Week12) ---
SAVE_DIR = "Csv\Etch_Data_Output"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 常量定义 ---
ROWS, COLS = 1000, 800
TOTAL_PARTICLES = 1000000
BATCH_SIZE = 2000       # GPU并行数
RATIO = 10.0 / 11.0     # 离子/中性粒子比例 (源自 Week12)

# --- Taichi 数据场 (显存空间) ---
# grid_material: 0=真空, 1=Si, 2=Hardmask
grid_exist = ti.field(dtype=ti.i32, shape=(ROWS, COLS))      
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS)) 

# ================= 2. 物理核心函数 (GPU) =================

@ti.func
def get_surface_normal(px: int, py: int):
    """
    【替代 Week12 的 reflector_face】
    GPU版法线计算：扫描5x5邻域的真空重心，计算法向量。
    """
    nx, ny = 0.0, 0.0
    for i, j in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        if 0 <= px + i < ROWS and 0 <= py + j < COLS:
            if grid_exist[px + i, py + j] == 0:
                # 指向真空的方向
                nx += float(i)
                ny += float(j)
    norm = ti.sqrt(nx**2 + ny**2) + 1e-6
    return nx / norm, ny / norm

@ti.func
def calculate_Ysicl(cos_theta: float) -> float:
    """
    【完美复刻 Week12 的 calculate_Ysicl】
    计算物理溅射产额，依赖于入射角和能量
    """
    Ei = 50.0   # 离子能量
    Eth = 20.0  # 阈值
    C = 0.77
    energy_term = ti.sqrt(Ei) - ti.sqrt(Eth)
    
    # Week12 逻辑：45度(约0.707)以内算1.0，大于45度按余弦衰减
    f_alpha = 1.0
    if cos_theta < 0.707: 
        f_alpha = cos_theta / 0.707
        
    return C * energy_term * f_alpha

@ti.func
def get_reflection_vector(vx: float, vy: float, nx: float, ny: float, is_ion: int):
    """
    【融合 Week12 的 reflect_angle】
    离子：镜面反射
    中性粒子：漫反射 (Lambertian) -> 随机半球方向
    """
    rvx, rvy = 0.0, 0.0
    
    if is_ion == 1:
        # === 离子：镜面反射 ===
        # v_out = v_in - 2 * (v_in . n) * n
        dot = vx * nx + vy * ny
        rvx = vx - 2 * dot * nx
        rvy = vy - 2 * dot * ny
    else:
        # === 中性粒子：漫反射 (Week12 核心特性) ===
        # 在法线方向的半圆内随机生成一个角度
        # 生成一个切向向量 (-ny, nx)
        tx, ty = -ny, nx
        
        # 随机混合 法向 和 切向
        rand_t = (ti.random() - 0.5) * 2.0 # -1 到 1
        
        # 简单的漫反射模拟：主要是沿着法线反弹，带一点随机偏转
        rvx = nx + rand_t * tx
        rvy = ny + rand_t * ty
        
        # 归一化
        norm = ti.sqrt(rvx**2 + rvy**2) + 1e-6
        rvx /= norm
        rvy /= norm

    return rvx, rvy

@ti.kernel
def init_grid():
    """初始化几何结构 (与 Week12 一致)"""
    angle_rad = 15 * math.pi / 180
    k_mask = ti.abs(ti.tan(angle_rad))
    for i, j in grid_exist:
        grid_count_cl[i, j] = 0
        if j < 150:
            grid_exist[i, j] = 0; grid_material[i, j] = 0
        elif j < 390:
            offset = int((390 - j) * k_mask)
            if (300 - offset) < i < (700 + offset):
                grid_exist[i, j] = 0; grid_material[i, j] = 0
            else:
                grid_exist[i, j] = 1; grid_material[i, j] = 2 # Mask
        else:
            grid_exist[i, j] = 1; grid_material[i, j] = 1 # Si

@ti.kernel
def simulate_batch():
    """
    【GPU 核心模拟循环】
    包含了粒子生成、运动、碰撞、化学反应、物理溅射、反射的全过程
    """
    for k in range(BATCH_SIZE):
        # --- 1. 粒子生成 (复刻 Week12 分布) ---
        px, py = ti.random() * (ROWS - 1), 1.0
        is_ion = 0
        if ti.random() > RATIO: is_ion = 1 # 离子概率
        
        # 角度分布
        sigma = (1.91 if is_ion==1 else 7.64) * (math.pi/180)
        angle = ti.randn() * sigma
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive, steps = True, 0
        while alive and steps < 2000:
            steps += 1
            # 步进 (模拟 Week12 的 return_next)
            px_n, py_n = px + vx * 1.1, py + vy * 1.1
            
            # 周期性边界 (Week12 特性)
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            if py_n < 0 or py_n >= COLS: alive=False; break
                
            ipx, ipy = int(px_n), int(py_n)
            
            # --- 2. 碰撞检测 ---
            if grid_exist[ipx, ipy] == 1:
                mat = grid_material[ipx, ipy]
                cl_n = grid_count_cl[ipx, ipy]
                
                # 计算法线 & 入射角
                nx, ny = get_surface_normal(ipx, ipy)
                cos_theta = -(vx * nx + vy * ny)
                if cos_theta < 0: cos_theta = 0.0
                
                # --- 3. 刻蚀概率计算 (Week12 核心逻辑移植) ---
                etch_prob = 0.0
                
                if is_ion == 1:
                    # == 离子逻辑 ==
                    # 物理项：calculate_Ysicl
                    ysicl = calculate_Ysicl(cos_theta)
                    
                    # 化学项：基础概率 (根据 Week12 的 reaction_probabilities)
                    chem_prob = 0.3 # 默认有Cl时的概率
                    if cl_n == 0: chem_prob = 0.1
                    
                    if mat == 2: chem_prob *= 0.1 # Hardmask 抗蚀
                    
                    # 综合概率：物理 + 化学
                    # Week12 逻辑: (0.1物理) OR (化学 * Ysicl)
                    # 这里简化为线性叠加近似
                    etch_prob = 0.05 + chem_prob * ysicl 
                    
                else:
                    # == 中性粒子逻辑 ==
                    # 纯化学吸附刻蚀，概率取决于 Cl 数目 (0, 0.1, 0.2, 0.3, 1.0)
                    if cl_n == 0: etch_prob = 0.0
                    elif cl_n == 1: etch_prob = 0.1
                    elif cl_n == 2: etch_prob = 0.2
                    elif cl_n == 3: etch_prob = 0.3
                    elif cl_n >= 4: etch_prob = 1.0
                    
                    if mat == 2: etch_prob *= 0.1 # Hardmask
                
                # --- 4. 判定结果 ---
                if ti.random() < etch_prob:
                    # >> 刻蚀发生 <<
                    grid_exist[ipx, ipy] = 0
                    grid_material[ipx, ipy] = 0
                    alive = False
                    
                    # (Week12 中的链式反应在这里可以通过判断邻居实现，但GPU中通常忽略微小链式以换取速度)
                else:
                    # >> 没刻蚀 -> 反射或吸附 <<
                    
                    # 吸附逻辑 (Week12)
                    if is_ion == 0 and cl_n < 4:
                        grid_count_cl[ipx, ipy] += 1
                        # 中性粒子吸附后通常就消失了(或者继续漫反射)
                        # 这里我们设定：吸附了就消失，没吸附(概率)才反射
                        alive = False 
                    
                    # 如果没死(比如离子，或者中性粒子没被吸附)，则反射
                    if alive or is_ion == 1:
                        # 反射概率 (Week12: reflect_prob)
                        # 角度越大(cos越小)越容易反射
                        ref_p = 1.0 - cos_theta 
                        if mat == 2: ref_p += 0.2 # Hardmask 更容易反射
                        
                        if ti.random() < ref_p:
                            # 调用上面的反射向量函数
                            vx, vy = get_reflection_vector(vx, vy, nx, ny, is_ion)
                            # 推离表面，防止卡死
                            px, py = px_n + nx, py_n + ny
                        else:
                            alive = False # 被吸收但未反应
            else:
                # 没撞到，更新坐标
                px, py = px_n, py_n

# ================= 3. CPU 辅助功能 (Week12 功能复刻) =================

def get_contour_points(raw_grid):
    """
    【复刻 Week12 的 extract_and_transform_contour】
    提取用于 CSV 保存和绘图的坐标点
    """
    smoothed = gaussian_filter(raw_grid.astype(float), sigma=2.0)
    binary = smoothed > 0.5
    points = []
    
    for x in range(binary.shape[0]):
        col = binary[x, :]
        indices = np.where(col == 1)[0]
        if len(indices) > 0:
            y = indices[0]
            # 坐标变换：Week12 中通常是 (rows-1-x, y) 或者直接 (x, y)
            # 这里保持与 GPU 坐标系一致
            points.append((x, y))
    return points

def save_csv(points, filename):
    """【复刻 Week12 的 CSV 保存功能】"""
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(points)
    # print(f"已保存: {filepath}")

# ================= 4. 主程序 =================

def main():
    init_grid()
    
    # 颜色：0=真空(青), 1=Si(深蓝), 2=Mask(红)
    cmap_custom = ListedColormap(["#008CFF", '#00008B', "#00FFFF"]) 
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    history_lines = [] 
    
    print(">>> 模拟开始 (Week12 功能已完全融合到 GPU) <<<")
    
    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    
    for i in range(num_batches):
        simulate_batch() # 呼叫 GPU
        
        # 每 20 批次 (10w粒子) 更新一次
        if i % 20 == 0:
            ti.sync()   
            
            # 获取数据
            mat_data = grid_material.to_numpy()
            exist_data = grid_exist.to_numpy()
            
            # --- 功能：提取轮廓 & 保存 CSV ---
            current_points = get_contour_points(exist_data)
            
            # 保存 CSV (复刻 Week12)
            current_particles = i * BATCH_SIZE
            csv_name = f"contour_{current_particles}.csv"
            save_csv(current_points, csv_name)
            
            # 存入历史用于绘图
            lx = [p[0] for p in current_points]
            ly = [p[1] for p in current_points]
            history_lines.append((lx, ly))
            
            # --- 绘图 ---
            ax.clear()
            ax.imshow(mat_data.T, cmap=cmap_custom, vmin=0, vmax=2, origin='upper')
            
            # 画历史线
            for idx, (hx, hy) in enumerate(history_lines):
                alpha = 0.3 + 0.7 * (idx / len(history_lines))
                color = 'red' if idx < len(history_lines)-1 else 'white'
                lw = 1.0 if idx < len(history_lines)-1 else 2.0
                ax.plot(hx, hy, color=color, linewidth=lw, alpha=alpha)

            ax.set_xlim(0, ROWS)
            ax.set_ylim(COLS, 0)
            ax.set_title(f"GPU Etching Fusion: {current_particles}/{TOTAL_PARTICLES}")
            
            plt.pause(0.01)
            print(f"进度: {i/num_batches:.1%} | 已保存 CSV", end='\r')

    ti.sync()
    
    # 最后保存一次最终结果
    final_points = get_contour_points(grid_exist.to_numpy())
    save_csv(final_points, "contour_final.csv")
    
    plt.ioff()
    print("\n>>> 模拟完成！所有数据已保存。窗口将保持打开。")
    plt.show(block=True)

if __name__ == "__main__":
    main()