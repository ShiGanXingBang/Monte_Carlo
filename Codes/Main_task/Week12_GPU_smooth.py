import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter # 用于图像平滑
import time
import math

# 1. 初始化 Taichi
ti.init(arch=ti.gpu)

# 2. 常量与场定义
ROWS, COLS = 800, 700
BATCH_SIZE = 5000
TOTAL_PARTICLES = 1000000
RATIO = 10.0 / 11.0

grid_exist = ti.field(dtype=ti.i32, shape=(ROWS, COLS))
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))

# ================= 物理逻辑 (精简版) =================

@ti.func
def get_normal(px: int, py: int):
    nx, ny = 0.0, 0.0
    for i, j in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        if 0 <= px + i < ROWS and 0 <= py + j < COLS:
            if grid_exist[px + i, py + j] == 0:
                nx += float(i); ny += float(j)
    norm = ti.sqrt(nx**2 + ny**2) + 1e-6
    return nx / norm, ny / norm

@ti.kernel
def init_grid():
    k_mask = ti.abs(ti.tan(5 * math.pi / 90))
    for i, j in grid_exist:
        if j < 50: grid_exist[i, j] = 0
        elif j < 200:
            offset = int((200 - j) * k_mask)
            if (300 - offset) < i < (500 + offset): grid_exist[i, j] = 0
            else:
                grid_exist[i, j] = 1
                grid_material[i, j] = 2
        else:
            grid_exist[i, j] = 1
            grid_material[i, j] = 1

@ti.kernel
def simulate_batch():
    for k in range(BATCH_SIZE):
        px, py = ti.random() * (ROWS - 1), 1.0
        is_ion = ti.random() > RATIO
        angle = ti.randn() * ((1.91 if is_ion else 7.64) * (math.pi/180))
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive, steps = True, 0
        while alive and steps < 1500:
            steps += 1
            px_n, py_n = px + vx * 1.2, py + vy * 1.2
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            if py_n < 0 or py_n >= COLS: break
            
            ipx, ipy = int(px_n), int(py_n)
            if grid_exist[ipx, ipy] == 1:
                # 简化版刻蚀判定逻辑
                nx, ny = get_normal(ipx, ipy)
                cos_theta = -(vx * nx + vy * ny)
                prob = 0.2 if is_ion else (grid_count_cl[ipx, ipy] * 0.25)
                
                if ti.random() < prob:
                    grid_exist[ipx, ipy] = 0
                    alive = False
                else:
                    # 反射
                    dot = vx * nx + vy * ny
                    vx, vy = vx - 2 * dot * nx, vy - 2 * dot * ny
                    px, py = px_n + nx, py_n + ny
            else:
                px, py = px_n, py_n

# ================= 实时平滑显示逻辑 =================

def run_simulation():
    init_grid()
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    
    for i in range(num_batches):
        simulate_batch()
        
        # 每 20 个 Batch 更新一次显示
        if i % 20 == 0:
            ti.sync()
            raw_img = grid_exist.to_numpy().T # 获取原始数据
            
            # --- 图像平滑处理 ---
            # sigma 越大越平滑，1.5-2.0 比较合适
            smooth_img = gaussian_filter(raw_img.astype(float), sigma=1.5)
            # 重新二值化，消除毛刺
            binary_smooth = smooth_img > 0.5 
            
            # 左图：原始数据 (带噪声/锯齿)
            ax1.clear()
            ax1.imshow(raw_img, cmap='gray')
            ax1.set_title("Original (Taichi Data)")
            
            # 右图：平滑后 (图像处理)
            ax2.clear()
            ax2.imshow(binary_smooth, cmap='Blues')
            ax2.set_title("Real-time Smoothed (Gaussian)")
            
            plt.pause(0.01)
            print(f"Progress: {i/num_batches:.1%}", end='\r')

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()