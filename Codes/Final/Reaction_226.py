import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
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
ti.init(arch=ti.gpu)

SAVE_DIR = r"Csv\Test_MarchingSquares_2026"
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

TOTAL_PARTICLES = 10000000
BATCH_SIZE = 4000
RATIO = 10.0 / 11.0  # 中性/总粒子比例

# --- Taichi 数据场 ---
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))      
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   
grid_temp = ti.field(dtype=ti.f32, shape=(ROWS, COLS))       

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
                        grid_material[x, y] = 2; grid_exist[x, y] = 1.0 # 
                
                for x in range(l_side, r_side):
                    if l_cur < x < r_cur:
                        grid_exist[x, y] = 0.0; grid_material[x, y] = 0
                    else:
                        grid_exist[x, y] = 1.0; grid_material[x, y] = 2
        
        last_r = current_right + int(Space/2)
        # 这里的ROWs绝对不能放成ROWs-1
        if last_r <= i < ROWS and vacuum < j < deep_border:
            grid_exist[i, j] = 1.0; grid_material[i, j] = 2
        if j >= deep_border and j < COLS:
            if grid_exist[i, j] == 0.0:
                grid_exist[i, j] = 1.0; grid_material[i, j] = 1

# ================= 4. 核心仿真逻辑 (反射优先) =================

@ti.kernel
def simulate_batch():
    for k in range(BATCH_SIZE):
        # --- A. 粒子生成 ---
        px, py = ti.random() * (ROWS - 1), 1.0
        
        is_ion = 0
        if ti.random() > RATIO: is_ion = 1
        
        sigma = (1.91 if is_ion==1 else 7.64) * (math.pi/180)
        angle = ti.randn() * sigma
        angle = max(min(angle, math.pi/2), -math.pi/2)
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive = True
        steps = 0
        ref_count = 0  # 反射计数
        
        while alive and steps < 3000:
            steps += 1
            # 移动
            px_n, py_n = px + vx * 1.1, py + vy * 1.1
            
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            if py_n < 0 or py_n >= COLS: alive = False; break
            
            ipx, ipy = int(px_n), int(py_n)
            
            # --- B. 碰撞检测 ---
            if grid_exist[ipx, ipy] > 0.5:
                mat = grid_material[ipx, ipy]
                cl_n = grid_count_cl[ipx, ipy]
                
                nx, ny = get_surface_normal(ipx, ipy)
                cos_theta = -(vx * nx + vy * ny)
                if cos_theta < 0: cos_theta = 0.0
                if cos_theta > 1: cos_theta = 1.0
                
                # 在 Taichi kernel 中使用 ti.acos，得到弧度制角度 (0..pi/2)
                theta_coll = ti.acos(cos_theta)

                # === 判定顺序：反射 -> 反应 ===
                
                did_reflect = False
                
                # --- 1. 反射判定 ---
                # 反射概率 (参考 reflect_prob 函数)
                # theta 对应 cos_theta，material=mat，species=is_ion
                threshold = math.pi / 3  # π/3
                prob_reflect = 0.0  # <--- 必须加上这一行初始化！

                if is_ion == 1:
                    #         # 当 cos_theta < π/3 时，反射概率从0线性增长到1
                    #         # 角度对应关系：theta = arccos(cos_theta)，需要将cos_theta转换为角度
                    #         # 简化：直接用 cos_theta 作为角度代理
                    angle_else = ti.max(0.0, (theta_coll - threshold) / (math.pi/2 - threshold))
                    angle_else = ti.min(1.0, angle_else)
                    prob_reflect = angle_else
                    # # 离子：掠角易反射 (1-cos)
                    # prob_reflect = 1.0 - cos_theta
                    if mat == 2: prob_reflect += 0.2
                    
                    if ti.random() < prob_reflect:
                        did_reflect = True
                        ref_count += 1
                else:
                    # 中性：粘附系数 (Cl越多越容易反弹)
                    prob_reflect = 0.5 + 0.1 * cl_n 
                    if prob_reflect > 0.95: prob_reflect = 0.95
                    #　实际上算的时候没有把这个跟黏着系数有关的模型加入，因为目前还没有找到相关的论文作为支撑。
                    if ti.random() < 0.8:
                        did_reflect = True
                        ref_count += 1
                
                # --- 2. 行为分支 ---
                if did_reflect:
                    # >> 反射物理 <<
                    vx, vy = get_reflection_vector(vx, vy, nx, ny, is_ion)
                    px = px + nx * 1.5
                    py = py + ny * 1.5
                else:
                    # >> 反应判定 (被捕获) <<
                    
                    if is_ion == 1:
                        # 离子刻蚀
                        prob_etch = 0.1 
                        if cl_n > 0: prob_etch += 0.2
                        
                        # [关键] 反射过的离子能量低，极难刻蚀
                        if ref_count >= 1:
                            prob_etch *= 0.1 
                        
                        if mat == 2: prob_etch *= 0.2 
                        
                        if ti.random() < prob_etch:
                            grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - 0.2)
                            if grid_exist[ipx, ipy] <= 0.0:
                                grid_material[ipx, ipy] = 0
                            alive = False 
                        else:
                            alive = False # 沉积/消失
                            
                    else:
                        # 中性粒子反应
                        prob_etch = 0.0
                        # 只有 Cl 饱和才刻蚀
                        if mat == 1 and cl_n >= 3:
                            prob_etch = 0.1
                        
                        if ti.random() < prob_etch:
                            # 刻蚀
                            grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - 0.2)
                            if grid_exist[ipx, ipy] <= 0.0:
                                grid_material[ipx, ipy] = 0
                                grid_count_cl[ipx, ipy] = 0
                            alive = False
                        else:
                            # 吸附
                            if cl_n < 5:
                                grid_count_cl[ipx, ipy] += 1
                            alive = False
                            
            else:
                px, py = px_n, py_n

# ================= 5. 轮廓提取 (元胞法 Marching Squares) =================

def get_contour_points(raw_grid):
    """
    【升级版】使用 Marching Squares (元胞法) 提取等值面轮廓
    """
    # 先做轻微高斯平滑，避免网格噪声导致轮廓破碎
    smoothed = gaussian_filter(raw_grid.astype(float), sigma=1.0)
    
    contour_points = []
    
    if HAS_SKIMAGE:
        # 方法 A: 使用 skimage.measure.find_contours (标准元胞法)
        # level=0.5 提取固液交界
        contours = measure.find_contours(smoothed, 0.5)
        
        # find_contours 返回的是 [(row, col), ...] 即 [(y, x)]
        # 我们需要把它展平并转换为 [(x, y)]
        
        # 只取最长的一条轮廓 (主结构)，或者是合并所有轮廓
        # 这里为了可视化方便，我们把所有轮廓点拼接起来
        for contour in contours:
            # 交换列(x)和行(y) -> (x, y)
            for p in contour:
                contour_points.append((p[1], p[0]))
                
    else:
        # 方法 B: 备用方案 (Matplotlib Contour Engine)
        # 如果没有 skimage，利用 matplotlib 的底层算法
        fig_temp = plt.figure()
        ax_temp = fig_temp.add_subplot(111)
        # 注意：contour 需要转置才能对应 (x, y)
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

# ================= 6. 主程序 =================

def main():
    if not HAS_SKIMAGE:
        print("Warning: 未检测到 scikit-image，将使用 Matplotlib 引擎提取轮廓。")
        print("建议安装: pip install scikit-image")

    init_grid()
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    history_lines = []
    
    print(">>> 模拟开始 (元胞法轮廓 + 反射优先逻辑) <<<")
    
    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    
    for i in range(num_batches):
        simulate_batch()
        smooth_grid()
        
        if i % 50 == 0: 
            ti.sync()
            
            mat_data = grid_material.to_numpy()
            exist_data = grid_exist.to_numpy()
            
            # --- 使用元胞法提取轮廓 ---
            points = get_contour_points(exist_data)
            
            current_count = i * BATCH_SIZE
            save_csv(points, f"contour_{current_count}.csv")
            
            # 提取 x, y 用于绘图 (注意：Marching Squares 的点是连续的，不需要排序)
            if len(points) > 0:
                # 为了画出连续线段，我们需要把它们按顺序存入
                # get_contour_points 返回的是点列表，通常已经是按顺序的片段
                # 这里我们简单地把它们拆分成 x 和 y
                lx = [p[0] for p in points]
                ly = [p[1] for p in points]
                
                # 为了防止不同闭环连在一起产生乱线，可以在这里做更复杂的处理
                # 但对于简单的沟槽结构，直接 append 通常没问题
                history_lines.append((lx, ly))
            
            # --- 绘图 ---
            ax.clear()
            
            rgb = np.zeros((ROWS, COLS, 3), dtype=np.float32)
            vac_mask = (exist_data < 0.5)
            mask_mask = (exist_data >= 0.5) & (mat_data == 2)
            si_mask = (exist_data >= 0.5) & (mat_data == 1)
            
            rgb[vac_mask] = to_rgb("#008CFF") 
            rgb[mask_mask] = to_rgb("#00FFFF") 
            rgb[si_mask] = to_rgb("#00008B")   
            
            ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='upper')
            
            # 绘制历史轮廓 (用 scatter 画点，或者 plot 画线)
            # 因为 Marching Squares 点很多，建议用细线
            for idx, (hx, hy) in enumerate(history_lines):
                alpha = 0.3 + 0.7 * (idx / len(history_lines))
                color = 'white' if idx == len(history_lines)-1 else 'red'
                # 使用 scatter 画点可能比 plot 连线更安全，防止首尾相连
                # 但用户要求"连线"，所以尝试 plot
                ax.plot(hy, hx, color=color, alpha=alpha, linewidth=1.0, linestyle='-')
                
            ax.set_title(f"Simulation: {current_count}/{TOTAL_PARTICLES}\nMethod: Marching Squares")
            ax.set_xlim(0, ROWS)
            ax.set_ylim(COLS, 0)
            
            plt.pause(0.01)
            print(f"进度: {i/num_batches:.1%}", end='\r')
            
    ti.sync()
    final_points = get_contour_points(grid_exist.to_numpy())
    save_csv(final_points, "contour_final.csv")
    print("\n>>> 模拟完成。")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()