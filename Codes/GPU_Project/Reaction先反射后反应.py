import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from scipy.ndimage import gaussian_filter
import math
import time
import os
import csv

# ================= 1. 环境初始化 =================
ti.init(arch=ti.gpu)

# --- 保存路径 ---
SAVE_DIR = r"Csv\Test_ReflectFirst_2026"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 全局常量 ---
ROWS, COLS = 1000, 700
vacuum = 100
deep_border = 230
left_border = 150
right_border = 250
Space = 100
Num = 3
CD = right_border - left_border

TOTAL_PARTICLES = 4000000
BATCH_SIZE = 4000
RATIO = 8.0 / 21.0  # 离子/中性粒子比例

# --- Taichi 数据场 ---
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))      # 0=空, 1=实
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   # 0=空, 1=Si, 2=Mask
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   # Cl 覆盖数
grid_temp = ti.field(dtype=ti.f32, shape=(ROWS, COLS))       # 平滑缓冲

# ================= 2. 物理辅助函数 =================

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
    """ 计算反射向量：离子=镜面，中性=漫反射 """
    rvx, rvy = 0.0, 0.0
    if is_ion == 1:
        # 镜面反射
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
    """ 网格平滑 (表面扩散模拟) """
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

# ================= 3. 几何初始化 =================

@ti.kernel
def init_grid():
    angle_rad = 5 * math.pi / 180
    k_mask = ti.abs(ti.tan(angle_rad))
    for i, j in grid_exist:
        grid_count_cl[i, j] = 0
        if j <= vacuum:
            grid_exist[i, j] = 0.0; grid_material[i, j] = 0
        
        current_left = left_border
        current_right = right_border
        for n in range(Num):
            if n == 0:
                current_left = left_border; current_right = right_border
            else:
                current_left += CD + Space; current_right += CD + Space
            
            for y in range(vacuum, deep_border):
                offset = int((deep_border - y) * k_mask)
                l_cur = max(0, min(current_left - offset, ROWS-1))
                r_cur = max(0, min(current_right + offset, ROWS-1))
                l_side = max(0, min(current_left - int(Space/2), ROWS-1))
                r_side = max(0, min(current_right + int(Space/2), ROWS-1))
                
                if n == 0:
                    for x in range(0, l_side):
                        grid_material[x, y] = 2; grid_exist[x, y] = 1.0
                
                for x in range(l_side, r_side):
                    if l_cur < x < r_cur:
                        grid_exist[x, y] = 0.0; grid_material[x, y] = 0
                    else:
                        grid_exist[x, y] = 1.0; grid_material[x, y] = 2
        
        last_r = current_right + int(Space/2)
        if last_r <= i < ROWS-1 and vacuum < j < deep_border:
            grid_exist[i, j] = 1.0; grid_material[i, j] = 2
        if j >= deep_border and j < COLS:
            if grid_exist[i, j] == 0.0:
                grid_exist[i, j] = 1.0; grid_material[i, j] = 1

# ================= 4. 核心仿真逻辑 (重写版) =================

@ti.kernel
def simulate_batch():
    for k in range(BATCH_SIZE):
        # --- A. 粒子生成 ---
        px, py = ti.random() * (ROWS - 1), 1.0
        
        is_ion = 0
        if ti.random() > RATIO: is_ion = 1
        
        # 角度分布
        sigma = (1.91 if is_ion==1 else 7.64) * (math.pi/180)
        angle = ti.randn() * sigma
        angle = max(min(angle, math.pi/2), -math.pi/2)
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive = True
        steps = 0
        ref_count = 0  # 反射计数器
        
        while alive and steps < 3000:
            steps += 1
            # 移动
            px_n, py_n = px + vx * 1.1, py + vy * 1.1
            
            # 边界
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            if py_n < 0 or py_n >= COLS: alive = False; break
            
            ipx, ipy = int(px_n), int(py_n)
            
            # --- B. 碰撞检测 ---
            if grid_exist[ipx, ipy] > 0.5:
                
                mat = grid_material[ipx, ipy]
                cl_n = grid_count_cl[ipx, ipy]
                
                # 计算几何信息
                nx, ny = get_surface_normal(ipx, ipy)
                cos_theta = -(vx * nx + vy * ny)
                if cos_theta < 0: cos_theta = 0.0
                if cos_theta > 1: cos_theta = 1.0
                
                # ==========================================
                #      逻辑核心：先判断反射，再判断反应
                # ==========================================
                
                did_reflect = False
                
                # --- 1. 判断反射 (Reflection Check) ---
                if is_ion == 1:
                    # [离子逻辑]
                    # 概率取决于入射角 (掠角易反射)
                    # 简单模型：prob = (1 - cos_theta)
                    prob_reflect = 1.0 - cos_theta
                    if mat == 2: prob_reflect += 0.2 # Mask 更易反射
                    
                    if ti.random() < prob_reflect:
                        # >> 发生反射 <<
                        did_reflect = True
                        ref_count += 1
                        
                else:
                    # [中性粒子逻辑]
                    # 概率取决于材料粘附系数 (Sticking Coeff)
                    # 假设 Cl 越多越难粘附 (反射率越高)
                    # base_reflect = 0.5 (假设)
                    prob_reflect = 0.5 + 0.1 * cl_n 
                    if prob_reflect > 0.95: prob_reflect = 0.95
                    
                    if ti.random() < prob_reflect:
                        # >> 发生反射 <<
                        did_reflect = True
                        # 中性粒子不限制反射次数，ref_count 仅用于统计或无用
                        ref_count += 1
                
                # --- 2. 执行逻辑分岔 ---
                if did_reflect:
                    # >> 执行反射物理 <<
                    vx, vy = get_reflection_vector(vx, vy, nx, ny, is_ion)
                    # 推离墙壁
                    px = px + nx * 1.5
                    py = py + ny * 1.5
                    # 活着，继续下一帧循环
                    
                else:
                    # >> 未反射 -> 进入反应判定 (Reaction Check) <<
                    # 此时粒子已被表面捕获，准备判定是 刻蚀 还是 沉积/消亡
                    
                    if is_ion == 1:
                        # == 离子反应 ==
                        # 基础刻蚀概率
                        prob_etch = 0.1 
                        # 如果有 Cl，概率增加 (化学辅助)
                        if cl_n > 0: prob_etch += 0.2
                        
                        # 【关键要求】如果反射过 (ref_count >= 1)，反应概率降低
                        if ref_count >= 1:
                            prob_etch *= 0.1 # 能量损失，很难刻蚀
                        
                        if mat == 2: prob_etch *= 0.4 # Mask 极难刻蚀
                        
                        if ti.random() < prob_etch:
                            # -> 刻蚀
                            grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - 0.2)
                            if grid_exist[ipx, ipy] <= 0.0:
                                grid_material[ipx, ipy] = 0
                            alive = False # 消耗
                        else:
                            # -> 未刻蚀 (且未反射)，视为沉积或能量耗尽
                            alive = False 
                            
                    else:
                        # == 中性粒子反应 ==
                        # 判定：刻蚀 vs 吸附
                        # 只有 Cl 够多 (>=3) 才能刻蚀 Si
                        prob_etch = 0.0
                        if mat == 1 and cl_n >= 3:
                            prob_etch = 0.5
                        
                        if ti.random() < prob_etch:
                            # -> 刻蚀 (挥发性产物)
                            grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - 0.2)
                            if grid_exist[ipx, ipy] <= 0.0:
                                grid_material[ipx, ipy] = 0
                                grid_count_cl[ipx, ipy] = 0 # 表面被挖走，Cl 清零
                            alive = False
                        else:
                            # -> 吸附 (增加 Cl)
                            if cl_n < 5:
                                grid_count_cl[ipx, ipy] += 1
                            alive = False
                            
            else:
                # 没撞到，继续飞
                px, py = px_n, py_n

# ================= 5. CPU 辅助 (轮廓提取 & 绘图) =================

def get_contour_points(raw_grid):
    smoothed = gaussian_filter(raw_grid.astype(float), sigma=2.0)
    binary = smoothed > 0.5
    points = []
    for x in range(binary.shape[0]):
        col = binary[x, :]
        indices = np.where(col == 1)[0]
        if len(indices) > 0:
            points.append((x, indices[0]))
    return points

def save_csv(points, filename):
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(points)

# ================= 6. 主程序 =================

def main():
    init_grid()
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    history_lines = []
    
    print(">>> 模拟开始 (Reflect-First Logic) <<<")
    
    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    
    for i in range(num_batches):
        simulate_batch()
        smooth_grid()
        
        if i % 50 == 0: # 每 20w 粒子刷新一次
            ti.sync()
            
            # --- 数据获取 ---
            mat_data = grid_material.to_numpy()
            exist_data = grid_exist.to_numpy()
            
            # --- 轮廓保存 ---
            points = get_contour_points(exist_data)
            current_count = i * BATCH_SIZE
            save_csv(points, f"contour_{current_count}.csv")
            
            lx = [p[0] for p in points]
            ly = [p[1] for p in points]
            history_lines.append((lx, ly))
            
            # --- 绘图 ---
            ax.clear()
            
            # 1. 背景填色 (构造 RGB)
            rgb = np.zeros((ROWS, COLS, 3), dtype=np.float32)
            vac_mask = (exist_data < 0.5)
            mask_mask = (exist_data >= 0.5) & (mat_data == 2)
            si_mask = (exist_data >= 0.5) & (mat_data == 1)
            
            rgb[vac_mask] = to_rgb("#008CFF")  # 真空蓝
            rgb[mask_mask] = to_rgb("#00FFFF") # Mask 青
            rgb[si_mask] = to_rgb("#00008B")   # Si 深蓝
            
            ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='upper')
            
            # 2. 历史轮廓
            for idx, (hx, hy) in enumerate(history_lines):
                alpha = 0.3 + 0.7 * (idx / len(history_lines))
                color = 'white' if idx == len(history_lines)-1 else 'red'
                lw = 2.0 if idx == len(history_lines)-1 else 1.0
                ax.plot(hx, hy, color=color, alpha=alpha, linewidth=lw)
                
            ax.set_title(f"Simulation: {current_count}/{TOTAL_PARTICLES}\nReflect First Logic")
            ax.set_xlim(0, ROWS)
            ax.set_ylim(COLS, 0)
            
            plt.pause(0.01)
            print(f"进度: {i/num_batches:.1%}", end='\r')
            
    # 结束
    ti.sync()
    final_points = get_contour_points(grid_exist.to_numpy())
    save_csv(final_points, "contour_final.csv")
    print("\n>>> 模拟完成。")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()