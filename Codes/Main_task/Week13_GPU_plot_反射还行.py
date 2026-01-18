import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap # 用于自定义颜色
from scipy.ndimage import gaussian_filter
import time
import math

# 1. 初始化 Taichi
ti.init(arch=ti.gpu)

# 2. 常量定义
ROWS, COLS = 800, 700
TOTAL_PARTICLES = 2000000
BATCH_SIZE = 5000 
RATIO = 20.0 / 21.0 # 离子比例

# 3. 数据场定义
# grid_material: 0=Vacuum, 1=Si, 2=Hardmask
grid_exist = ti.field(dtype=ti.i32, shape=(ROWS, COLS))      
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS)) 

# ================= 物理内核 (保持不变) =================

@ti.func
def get_surface_normal(px: int, py: int):
    nx, ny = 0.0, 0.0
    for i, j in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        if 0 <= px + i < ROWS and 0 <= py + j < COLS:
            if grid_exist[px + i, py + j] == 0:
                nx += float(i); ny += float(j)
    norm = ti.sqrt(nx**2 + ny**2) + 1e-6
    return nx / norm, ny / norm

@ti.func
def get_ysicl_factor(cos_theta: float) -> float:
    f_alpha = 1.0
    if cos_theta < 0.707: f_alpha = cos_theta / 0.707
    return 0.77 * (ti.sqrt(50.0) - ti.sqrt(20.0)) * f_alpha

@ti.kernel  
def init_grid():
    angle_rad = 5 * math.pi / 180
    k_mask = ti.abs(ti.tan(angle_rad))
    for i, j in grid_exist:
        grid_count_cl[i, j] = 0
        if j < 50:
            grid_exist[i, j] = 0; grid_material[i, j] = 0 # Vacuum
        elif j < 200:
            offset = int((200 - j) * k_mask)
            if (300 - offset) < i < (500 + offset):
                grid_exist[i, j] = 0; grid_material[i, j] = 0 # Vacuum
            else:
                grid_exist[i, j] = 1; grid_material[i, j] = 2 # Mask
        else:
            grid_exist[i, j] = 1; grid_material[i, j] = 1 # Si

@ti.kernel
def simulate_batch():
    for k in range(BATCH_SIZE):
        px, py = ti.random() * (ROWS - 1), 1.0
        is_ion = ti.random() > RATIO 
        sigma = (1.91 if is_ion else 7.64) * (math.pi/180)
        angle = ti.randn() * sigma
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive, steps = True, 0
        while alive and steps < 2000:
            steps += 1
            px_n, py_n = px + vx * 1.1, py + vy * 1.1
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            if py_n < 0 or py_n >= COLS: alive=False; break
                
            ipx, ipy = int(px_n), int(py_n)
            
            if grid_exist[ipx, ipy] == 1:
                mat = grid_material[ipx, ipy]
                cl_n = grid_count_cl[ipx, ipy]
                nx, ny = get_surface_normal(ipx, ipy)
                cos_theta = -(vx * nx + vy * ny)
                if cos_theta < 0: cos_theta = 0.0
                
                etch_prob = 0.0
                if is_ion:
                    if mat == 1: etch_prob = 0.1 + 0.1 * get_ysicl_factor(cos_theta)
                    else: etch_prob = 0.01
                else:
                    if cl_n == 1: etch_prob = 0.1
                    elif cl_n == 2: etch_prob = 0.2
                    elif cl_n == 3: etch_prob = 0.3
                    elif cl_n == 4: etch_prob = 1.0
                    if mat == 2: etch_prob *= 0.1
                
                if ti.random() < etch_prob:
                    grid_exist[ipx, ipy] = 0
                    grid_material[ipx, ipy] = 0 # 变成真空
                    alive = False
                else:
                    if ti.random() < (1.0 - cos_theta):
                        dot = vx * nx + vy * ny
                        vx, vy = vx - 2 * dot * nx, vy - 2 * dot * ny
                        px, py = px_n + nx, py_n + ny
                    else:
                        if not is_ion and cl_n < 4: grid_count_cl[ipx, ipy] += 1
                        alive = False
            else:
                px, py = px_n, py_n

# ================= 辅助函数：提取平滑轮廓线 =================

def get_smooth_contour_line(raw_grid):
    """
    输入：原始的0/1网格 (numpy array)
    输出：平滑后的轮廓线坐标 (x_list, y_list)
    """
    # 1. 高斯平滑
    smoothed = gaussian_filter(raw_grid.astype(float), sigma=2.0)
    # 2. 阈值化
    binary = smoothed > 0.5
    
    # 3. 提取每列的最上方表面点
    # binary shape is (ROWS, COLS)
    line_x = []
    line_y = []
    
    for x in range(binary.shape[0]):
        # 在该列中找第一个非0点（即第一个材料点）
        # 注意：由于y=0是顶部，我们找的是第一个1
        col = binary[x, :]
        indices = np.where(col == 1)[0]
        
        if len(indices) > 0:
            y_surface = indices[0]
            line_x.append(x)
            line_y.append(y_surface)
            
    return line_x, line_y

# ================= 主程序 =================

def main():
    init_grid()
    
    # --- 1. 设置颜色映射 ---
    # 0: Vacuum (蓝色), 1: Si (深蓝), 2: Mask (浅蓝)
    # 颜色顺序对应数值 0, 1, 2
    colors = ["#008CFF", '#00008B', "#00FFFF"] 
    cmap_custom = ListedColormap(colors)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    start_t = time.time()
    history_lines = [] # 用于存储历史轮廓
    
    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    
    print("开始模拟...")
    
    for i in range(num_batches):
        simulate_batch()
        
        # 每 20 个 batch (约10万粒子) 记录并更新一次
        if i % 20 == 0:
            ti.sync()
            
            # 获取数据
            mat_data = grid_material.to_numpy() # 用于画颜色 (0,1,2)
            exist_data = grid_exist.to_numpy()  # 用于算轮廓 (0,1)
            
            # --- 记录历史轮廓 ---
            # 提取当前的平滑轮廓线
            lx, ly = get_smooth_contour_line(exist_data)
            # 存入历史列表，附带当前进度信息(可选)
            history_lines.append((lx, ly))
            
            # --- 绘图 ---
            ax.clear()
            
            # 1. 画背景材质 (不同颜色)
            # 转置(.T)是因为 matplotlib 和 taichi 的行列定义差异
            # vmin=0, vmax=2 确保颜色映射正确对应 0,1,2
            ax.imshow(mat_data.T, cmap=cmap_custom, vmin=0, vmax=2, origin='upper')
            
            # 2. 画历史轮廓线
            # 遍历历史记录，画出浅色的线
            for idx, (hx, hy) in enumerate(history_lines):
                # 越新的线越不透明，越旧的越透明
                alpha = 0.3 + 0.7 * (idx / len(history_lines))
                # 倒数第二个之前的都是历史线，设为红色
                if idx < len(history_lines) - 1:
                    ax.plot(hx, hy, color='red', linewidth=0.8, alpha=0.4)
                else:
                    # 最新的一条线，画粗一点，白色
                    ax.plot(hx, hy, color='white', linewidth=1.5, alpha=1.0)

            # 设置显示范围和标签
            ax.set_xlim(0, ROWS)
            ax.set_ylim(COLS, 0) # Y轴翻转，0在上面
            ax.set_title(f"GPU Etching: {i*BATCH_SIZE}/{TOTAL_PARTICLES} particles")
            
            # 图例说明 (手动添加)
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#008CFF', label='Vacuum'),
                Patch(facecolor='#00008B', label='Si (Silicon)'),
                Patch(facecolor='#00FFFF', label='Mask'),
                plt.Line2D([0], [0], color='red', lw=1, label='History'),
                plt.Line2D([0], [0], color='white', lw=1.5, label='Current')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.pause(0.01)
            print(f"Progress: {i/num_batches:.1%}", end='\r')

    ti.sync()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()