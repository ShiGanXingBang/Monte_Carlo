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
ti.init(arch=ti.gpu)

# --- 保存路径设置 ---
SAVE_DIR = "Csv/TEST2026.1.15_Smooth_Etch"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 常量定义 ---
ROWS, COLS = 1000, 700
vacuum = 100
deep_border = 230
left_border = 150
right_border = 250
Space = 100 
Num = 3
CD = right_border - left_border 

TOTAL_PARTICLES = 5000000
BATCH_SIZE = 2000
RATIO = 20.0 / 21.0 

# --- 【修改点1】Taichi 数据场升级 ---
# grid_exist 改为 f32 (浮点数)，用于存储“密度/血量” (0.0 ~ 1.0)
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))      
# 新增一个临时场用于平滑计算，防止读写冲突
grid_temp  = ti.field(dtype=ti.f32, shape=(ROWS, COLS))

grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS)) 

# ================= 2. 物理核心函数 (GPU) =================

@ti.func
def get_surface_normal(px: int, py: int):
    """
    计算法线：扫描5x5邻域。
    【修改点】判定真空的标准从 ==0 变为 <0.5
    """
    nx, ny = 0.0, 0.0
    for i, j in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        if 0 <= px + i < ROWS and 0 <= py + j < COLS:
            # 如果邻居是真空 (密度小于0.5)，则法线倾向于该方向
            if grid_exist[px + i, py + j] < 0.5:
                nx += float(i)
                ny += float(j)
    norm = ti.sqrt(nx**2 + ny**2) + 1e-6
    return nx / norm, ny / norm

@ti.func
def calculate_Ysicl(cos_theta: float) -> float:
    Ei = 50.0
    Eth = 20.0
    C = 0.77
    energy_term = ti.sqrt(Ei) - ti.sqrt(Eth)
    f_alpha = 1.0
    if cos_theta < 0.707: 
        f_alpha = cos_theta / 0.707
    return C * energy_term * f_alpha

@ti.func
def get_reflection_vector(vx: float, vy: float, nx: float, ny: float, is_ion: int):
    rvx, rvy = 0.0, 0.0
    if is_ion == 1:
        dot = vx * nx + vy * ny
        rvx = vx - 2 * dot * nx
        rvy = vy - 2 * dot * ny
    else:
        # === 中性粒子：真正的漫反射 ===
        # 1. 先计算法线的绝对角度 (使用 atan2)
        angle_normal = ti.atan2(ny, nx)
        
        # 2. 在 -90度 到 +90度 (即 -PI/2 到 +PI/2) 之间生成一个随机偏转角
        # ti.random() -> [0, 1]
        # (ti.random() - 0.5) -> [-0.5, 0.5]
        # * ti.math.pi -> [-PI/2, PI/2]
        angle_deviation = (ti.random() - 0.5) * 3.1415926
        
        # 3. 叠加角度得到新的飞行方向
        theta_new = angle_normal + angle_deviation
        
        # 4. 转换回向量
        rvx = ti.cos(theta_new)
        rvy = ti.sin(theta_new)
        
        # (此时 rvx, rvy 已经是单位向量，不需要归一化，且肯定指向真空侧)
    return rvx, rvy

# --- 【新增功能】网格平滑内核 ---
@ti.kernel
def smooth_grid():
    """
    对 grid_exist 进行轻微的高斯平滑/扩散。
    这会让锯齿边缘变得平滑。
    """
    # 权重配置：中心保持 98%，周围分 2%。这样平滑很微弱，不会糊掉。
    w_center = 0.98
    w_neighbor = (1.0 - w_center) / 4.0
    
    for i, j in grid_exist:
        # 只处理非边界区域
        if 1 <= i < ROWS - 1 and 1 <= j < COLS - 1:
            val = (grid_exist[i, j] * w_center +
                   (grid_exist[i+1, j] + grid_exist[i-1, j] + 
                    grid_exist[i, j+1] + grid_exist[i, j-1]) * w_neighbor)
            grid_temp[i, j] = val
        else:
            grid_temp[i, j] = grid_exist[i, j]
            
    # 将平滑后的结果写回
    for i, j in grid_exist:
        grid_exist[i, j] = grid_temp[i, j]

@ti.kernel
def init_grid():
    """初始化几何结构 (适配 float 类型)"""
    angle_rad = 5 * math.pi / 180
    k_mask = ti.abs(ti.tan(angle_rad))
    for i, j in grid_exist:
        grid_count_cl[i, j] = 0
        if j <= vacuum:
            # 【修改点】赋值为 0.0
            grid_exist[i, j] = 0.0; grid_material[i, j] = 0

        current_left_border = left_border
        current_right_border = right_border

        for n in range(Num):
            if n == 0:
                current_left_border = left_border 
                current_right_border = right_border
            else:
                current_left_border = current_left_border + CD + Space
                current_right_border = current_right_border + CD + Space

            for y in range(vacuum, deep_border): 
                offset = int((deep_border - y) * k_mask)
                left_current = current_left_border - offset
                right_current = current_right_border + offset
                left_current = max(0, min(left_current, ROWS - 1))
                right_current = max(0, min(right_current, ROWS - 1))
                left_side = max(0, min(current_left_border - int(Space / 2), ROWS - 1))
                right_side = max(0, min(current_right_border + int(Space / 2), ROWS - 1))
                
                if n == 0:
                    for x in range(0, left_side):
                        grid_material[x, y] = 2
                        grid_exist[x, y] = 1.0 # 【修改点】1.0
                
                for x in range(left_side, right_side):
                    if  left_current < x < right_current:
                        grid_exist[x, y] = 0.0; grid_material[x, y] = 0
                    else:
                        grid_exist[x, y] = 1.0; grid_material[x, y] = 2 
        
        # 修正右侧边界逻辑
        last_right_side = current_right_border + int(Space / 2)
        if last_right_side - 1 < i < ROWS and vacuum < j < deep_border:
             grid_exist[i, j] = 1.0; grid_material[i, j] = 2
                
        if grid_exist[i, j] == 0.0 and j >= deep_border:
             grid_exist[i, j] = 1.0; grid_material[i, j] = 1 

@ti.kernel
def simulate_batch():
    for k in range(BATCH_SIZE):
        px, py = ti.random() * (ROWS - 1), 1.0
        is_ion = 0
        if ti.random() > RATIO: is_ion = 1
        
        sigma = (1.91 if is_ion==1 else 7.64) * (math.pi/180)
        angle = ti.randn() * sigma
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive, steps = True, 0
        
        # 【修改点：反射限制】 初始化计数器
        ref_count = 0 
        
        while alive and steps < 2000:
            steps += 1
            px_n, py_n = px + vx * 1.1, py + vy * 1.1
            
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            if py_n < 0 or py_n >= COLS: alive=False; break
                
            ipx, ipy = int(px_n), int(py_n)
            
            # --- 碰撞检测 ---
            # 【修改点】只要密度 > 0.5 就视为撞击
            if grid_exist[ipx, ipy] > 0.5:
                mat = grid_material[ipx, ipy]
                cl_n = grid_count_cl[ipx, ipy]
                
                nx, ny = get_surface_normal(ipx, ipy)
                cos_theta = -(vx * nx + vy * ny)
                if cos_theta < 0: cos_theta = 0.0
                
                # --- 刻蚀概率 ---
                etch_prob = 0.0
                if is_ion == 1:
                    ysicl = calculate_Ysicl(cos_theta)
                    chem_prob = 0.3
                    if cl_n == 0: chem_prob = 0.1
                    elif cl_n == 1: chem_prob = 0.3 * 0.25
                    elif cl_n == 2: chem_prob = 0.3 * 0.25 * 2
                    elif cl_n == 3: chem_prob = 0.3 * 0.25 * 3
                    elif cl_n >= 4: chem_prob = 0.3 * 0.25 * 4
                    if mat == 2: chem_prob *= 0.4
                    etch_prob = 0.05 + chem_prob * ysicl 
                else:
                    if cl_n == 3: etch_prob = 0.1
                    if mat == 2: etch_prob *= 0.4
                
                if ti.random() < etch_prob:
                    # >> 刻蚀发生 <<
                    # 【修改点】渐进式刻蚀，每次减去 0.2 (相当于5次撞击挖空一个格)
                    # 这样可以利用平滑后的数值边界
                    grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - 0.2)
                    
                    # 如果彻底挖空了，把材料标记也清掉（可选，方便调试）
                    if grid_exist[ipx, ipy] <= 0.0:
                        grid_material[ipx, ipy] = 0
                        
                    alive = False
                else:
                    # >> 吸附/反射 <<
                    Prob_next = 0.0
                    if cl_n == 0: Prob_next = 1.00
                    elif cl_n == 1: Prob_next = 0.75
                    elif cl_n == 2: Prob_next = 0.50
                    elif cl_n == 3: Prob_next = 0.25

                    if is_ion == 0 and cl_n < 4 and ti.random() < Prob_next:
                        grid_count_cl[ipx, ipy] += 1
                        alive = False 
                    
                    if alive or is_ion == 1:
                        ref_p = 1.0 - cos_theta 
                        if mat == 2: ref_p += 0.2
                        
                        if ti.random() < ref_p:
                            # 【修改点：反射限制逻辑】
                            if ref_count >= 5: # 限制只反射1次
                                alive = False
                            else:
                                vx, vy = get_reflection_vector(vx, vy, nx, ny, is_ion)
                                px, py = px_n + nx, py_n + ny
                                ref_count += 1 # 增加计数
                        else:
                            alive = False
            else:
                px, py = px_n, py_n

# ================= 3. CPU 辅助功能 =================

def get_contour_points(raw_grid):
    # 这里 raw_grid 现在是 float，可以直接用
    # 取 0.5 作为固液界面
    binary = raw_grid > 0.5 
    points = []
    for x in range(binary.shape[0]):
        col = binary[x, :]
        indices = np.where(col == 1)[0]
        if len(indices) > 0:
            y = indices[0]
            points.append((x, y))
    return points

def save_csv(points, filename):
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(points)

# ================= 4. 主程序 =================

def main():
    init_grid()
    
    cmap_custom = ListedColormap(["#008CFF", '#00008B', "#00FFFF"]) 
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    history_lines = [] 
    
    print(">>> 模拟开始 (平滑刻蚀 + 单次反射限制) <<<")
    
    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    
    for i in range(num_batches):
        simulate_batch()
        
        # 【修改点】每批次结束后，进行一次微弱的平滑
        # 这就是让轮廓在刻蚀过程中保持圆润的关键！
        smooth_grid()
        
        if i % 100 == 0:
            ti.sync()
            mat_data = grid_material.to_numpy()
            exist_data = grid_exist.to_numpy() # 现在取出的是 float 数据
            
            current_points = get_contour_points(exist_data)
            
            # 保存
            current_particles = i * BATCH_SIZE
            csv_name = f"contour_{current_particles}.csv"
            save_csv(current_points, csv_name)
            
            lx = [p[0] for p in current_points]
            ly = [p[1] for p in current_points]
            history_lines.append((lx, ly))
            
            # 绘图
            ax.clear()
            # 用 exist_data 画背景可以看到灰度平滑效果
            ax.imshow(mat_data.T, cmap=cmap_custom, vmin=0, vmax=2, origin='upper')
            
            for idx, (hx, hy) in enumerate(history_lines):
                alpha = 0.3 + 0.7 * (idx / len(history_lines))
                color = 'red' if idx < len(history_lines)-1 else 'white'
                lw = 1.0 if idx < len(history_lines)-1 else 2.0
                ax.plot(hx, hy, color=color, linewidth=lw, alpha=alpha)

            ax.set_xlim(0, ROWS)
            ax.set_ylim(COLS, 0)
            ax.set_title(f"Smooth Etching: {current_particles}/{TOTAL_PARTICLES}")
            
            plt.pause(0.01)
            print(f"进度: {i/num_batches:.1%} | 已保存 CSV", end='\r')

    ti.sync()
    final_points = get_contour_points(grid_exist.to_numpy())
    save_csv(final_points, "contour_final.csv")
    
    plt.ioff()
    print("\n>>> 模拟完成！")
    plt.show(block=True)

if __name__ == "__main__":
    main()