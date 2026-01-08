import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
import math
import os
import csv
import time

ti.init(arch=ti.gpu)

# --- 配置与场定义 ---
ROWS, COLS = 800, 700
BATCH_SIZE = 5000
TOTAL_PARTICLES = 1000000 
RATIO = 10.0 / 11.0 # 10份中性粒子 : 1份离子

grid_exist = ti.field(dtype=ti.i32, shape=(ROWS, COLS))      
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS)) # 0:纯Si, 1:SiCl, 2:SiCl2, 3:SiCl3

# ================= 核心：反应机理函数 =================

@ti.func
def get_etch_probability(is_ion: int, cl_stage: int, material: int, cos_theta: float):
    """
    根据上传的机理图逻辑实现的概率模型
    is_ion: 1=离子, 0=中性粒子
    cl_stage: 当前格子的氯化程度 (0-3)
    material: 1=Si, 2=Hardmask
    """
    prob = 0.0
    
    if material == 1: # 硅层
        if is_ion == 1:
            # --- 1. 离子路径 ---
            # 物理溅射产额 (由Week12公式计算)
            Y_phys = 0.77 * (ti.sqrt(50.0) - ti.sqrt(20.0)) * (cos_theta / 0.707 if cos_theta < 0.707 else 1.0)
            
            if cl_stage == 0:
                # 纯物理溅射：概率较低
                prob = 0.1 * Y_phys
            else:
                # 离子辅助反应：离子撞击氯化表面，产额大幅提升
                # 随着 cl_stage 增加，反应变得极其容易 (接力去除)
                prob = 0.4 * Y_phys * (cl_stage + 1) 
        else:
            # --- 2. 中性粒子路径 (Cl原子) ---
            if cl_stage < 3:
                # 只是吸附过程，不产生刻蚀（返回负值代表吸附增加）
                prob = -1.0 
            else:
                # 自发反应：当达到 SiCl4 状态时(本处模拟为stage3再吸附)，极高概率自发挥发
                prob = 0.8 
                
    elif material == 2: # 掩膜层
        # 掩膜通常只考虑物理溅射且概率极低
        if is_ion == 1: prob = 0.01
        else: prob = 0.0
            
    return prob

# ================= 物理模拟内核 =================

@ti.kernel
def simulate_batch():
    for k in range(BATCH_SIZE):
        px, py = ti.random() * (ROWS - 1), 1.0
        is_ion = 1 if ti.random() > RATIO else 0
        
        # 初始角度分布 (Week12标准)
        angle = ti.randn() * ((1.91 if is_ion==1 else 7.64) * (math.pi/180))
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive = True
        steps = 0
        while alive and steps < 1500:
            steps += 1
            px_n, py_n = px + vx * 1.2, py + vy * 1.2
            
            # 周期性边界
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            if py_n < 0 or py_n >= COLS: break
            
            ipx, ipy = int(px_n), int(py_n)
            if grid_exist[ipx, ipy] == 1:
                # 碰撞发生，计算法线和入射角
                nx, ny = 0.0, 0.0 # 简易梯度法
                for i, j in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    if grid_exist[ipx+i, ipy+j] == 0: nx += i; ny += j
                norm = ti.sqrt(nx**2 + ny**2) + 1e-6
                nx /= norm; ny /= norm
                cos_theta = -(vx * nx + vy * ny)
                
                # 获取机理概率
                p = get_etch_probability(is_ion, grid_count_cl[ipx, ipy], grid_material[ipx, ipy], cos_theta)
                
                if p < 0: # 发生吸附
                    grid_count_cl[ipx, ipy] += 1
                    alive = False
                elif ti.random() < p: # 发生刻蚀（物理溅射/自发/离子辅助）
                    grid_exist[ipx, ipy] = 0
                    grid_material[ipx, ipy] = 0
                    alive = False
                else: # 没反应，发生反射 (Week12 镜面反射逻辑)
                    dot = vx * nx + vy * ny
                    vx, vy = vx - 2 * dot * nx, vy - 2 * dot * ny
                    px, py = px_n + nx, py_n + ny
            else:
                px, py = px_n, py_n

# ================= 绘图与功能模块 =================

def main():
    # 初始化网格逻辑 (同Week13)
    # ... (此处省略重复的 init_grid 代码以节省篇幅) ...
    # 记得在 init_grid 中加入 grid_count_cl[i,j]=0

    print(">>> 深度机理模型已启动...")
    # 使用自定义 Colormap 增强视觉效果
    # 0:真空(黑), 1:Si(深蓝), 2:Mask(金)
    my_cmap = ListedColormap(['#121212', '#1E90FF', '#FFD700'])
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 10))
    
    for b in range(TOTAL_PARTICLES // BATCH_SIZE):
        simulate_batch()
        if b % 25 == 0:
            ti.sync()
            img = grid_material.to_numpy().T
            # 这里不仅画材质，我们可以通过 grid_count_cl 叠加显示表面活性
            ax.clear()
            ax.imshow(img, cmap=my_cmap, origin='upper')
            ax.set_title(f"Mechanism Simulation: {b*BATCH_SIZE} particles")
            plt.pause(0.01)

# (此处补全 init_grid 并调用 main)
# ... (接上一段代码中的 simulate_batch 之后)

# ================= 3. 初始化网格逻辑 (复刻 Week12 的几何结构) =================

@ti.kernel
def init_grid():
    """
    初始化几何结构：
    1. 顶部 50 像素为真空
    2. 50-200 像素为带 30 度斜角的 Hardmask
    3. 200 像素以下为纯 Si
    """
    angle_rad = 30 * math.pi / 90  # 30度角
    k_mask = ti.abs(ti.tan(angle_rad))
    
    for i, j in grid_exist:
        grid_count_cl[i, j] = 0  # 初始所有格子的 Cl 吸附量为 0
        if j < 50:
            # 顶部真空区
            grid_exist[i, j] = 0
            grid_material[i, j] = 0
        elif j < 200:
            # 计算掩膜开口宽度 (倒梯形)
            offset = int((200 - j) * k_mask)
            if (300 - offset) < i < (500 + offset):
                # 开口部分为真空
                grid_exist[i, j] = 0
                grid_material[i, j] = 0
            else:
                # 掩膜部分
                grid_exist[i, j] = 1
                grid_material[i, j] = 2  # 2 代表 Hardmask
        else:
            # 底部硅基底
            grid_exist[i, j] = 1
            grid_material[i, j] = 1  # 1 代表 Si

# ================= 4. 辅助功能 (轮廓提取与 CSV 保存) =================

def get_contour_line(raw_grid):
    """提取每一列最顶端的材料点坐标，用于保存轮廓"""
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(raw_grid.astype(float), sigma=2.0)
    binary = smoothed > 0.5
    line_x, line_y = [], []
    for x in range(binary.shape[0]):
        indices = np.where(binary[x, :] == 1)[0]
        if len(indices) > 0:
            line_x.append(x)
            line_y.append(indices[0])
    return line_x, line_y

def save_contour_to_csv(lx, ly, batch_idx):
    """保存 CSV 数据到 Week12 指定的目录"""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    filename = os.path.join(SAVE_DIR, f"contour_step_{batch_idx}.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(zip(lx, ly))

# ================= 5. 主程序调用 (main) =================

def main():
    # 1. 初始化
    init_grid()
    
    # 2. 设置绘图美化 (可视化不同材质)
    # 0:真空(深灰色), 1:Si(深蓝色), 2:Hardmask(火砖红/橙色)
    colors = ['#1C1C1C', '#00008B', '#FF4500']
    my_cmap = ListedColormap(colors)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    history_lines = []
    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    
    print(f">>> 开始机理仿真：共计 {TOTAL_PARTICLES} 粒子")
    print(f">>> 数据保存路径: {SAVE_DIR}")

    start_time = time.time()

    for i in range(num_batches):
        # 执行 GPU 计算
        simulate_batch()
        
        # 每 20 个 Batch 更新一次 UI 和保存数据
        if i % 20 == 0:
            ti.sync()
            
            # 获取数据到 CPU
            mat_data = grid_material.to_numpy()
            exist_data = grid_exist.to_numpy()
            
            # 提取平滑轮廓
            lx, ly = get_contour_line(exist_data)
            history_lines.append((lx, ly))
            
            # 保存数据 (复刻 Week12 功能)
            save_contour_to_csv(lx, ly, i * BATCH_SIZE)
            
            # --- 绘图逻辑 ---
            ax.clear()
            # 画背景材质
            ax.imshow(mat_data.T, cmap=my_cmap, vmin=0, vmax=2, origin='upper')
            
            # 画历史轮廓 (模拟 Week12 的多轮廓叠加)
            for idx, (hx, hy) in enumerate(history_lines):
                alpha = 0.2 + 0.8 * (idx / len(history_lines))
                color = 'white' if idx == len(history_lines)-1 else 'yellow'
                ax.plot(hx, hy, color=color, linewidth=1, alpha=alpha)
            
            ax.set_title(f"Mechanism: {i*BATCH_SIZE} ions | Time: {time.time()-start_time:.1f}s")
            ax.set_xlim(0, ROWS)
            ax.set_ylim(COLS, 0)
            
            plt.pause(0.01)
            print(f"进度: {i/num_batches:.1%}", end='\r')

    # 模拟结束，保持窗口
    print("\n>>> 仿真圆满结束！所有数据已保存。")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # 配置保存路径 (请确保该路径在你电脑上存在)
    SAVE_DIR = "Etching_Results_Mechanism"
    main()