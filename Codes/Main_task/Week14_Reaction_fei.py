import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
import math
import os

ti.init(arch=ti.gpu) 

# --- 常量定义 ---
ROWS, COLS = 1000, 800
TOTAL_PARTICLES = 3000000
BATCH_SIZE = 2000
RATIO = 10.0 / 11.0 

# --- Taichi 数据场 ---
grid_exist = ti.field(dtype=ti.i32, shape=(ROWS, COLS))      
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS)) 

# ================= 1. 物理核心函数 =================

@ti.func
def get_surface_normal(px: int, py: int):
    nx, ny = 0.0, 0.0
    for i, j in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        if 0 <= px + i < ROWS and 0 <= py + j < COLS:
            if grid_exist[px + i, py + j] == 0:
                nx += float(i)
                ny += float(j)
    norm = ti.sqrt(nx**2 + ny**2) + 1e-6
    return nx / norm, ny / norm

@ti.func
def get_reflection_vector(vx: float, vy: float, nx: float, ny: float, is_ion: int):
    rvx, rvy = 0.0, 0.0
    if is_ion == 1:
        dot = vx * nx + vy * ny
        rvx = vx - 2 * dot * nx
        rvy = vy - 2 * dot * ny
    else:
        # 中性粒子漫反射
        tx, ty = -ny, nx
        rand_t = (ti.random() - 0.5) * 2.0
        rvx = nx + rand_t * tx
        rvy = ny + rand_t * ty
        norm = ti.sqrt(rvx**2 + rvy**2) + 1e-6
        rvx /= norm
        rvy /= norm
    return rvx, rvy

# ================= 2. 几何初始化 (完全还原 Week12) =================

@ti.kernel
def init_grid():
    """还原你要求的 15度 掩膜结构"""
    angle_rad = 5 * math.pi / 180
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

# ================= 3. 反应逻辑 (核心) =================

@ti.kernel
def simulate_batch():
    for k in range(BATCH_SIZE):
        px, py = ti.random() * (ROWS - 1), 1.0
        is_ion = 0
        if ti.random() > RATIO: is_ion = 1
        
        sigma = (1.91 if is_ion==1 else 45.0) * (math.pi/180)
        angle = ti.randn() * sigma
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive, steps = True, 0
        while alive and steps < 2000:
            steps += 1
            px_n, py_n = px + vx, py + vy
            
            # 边界及周期
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            if py_n < 0 or py_n >= COLS: alive=False; break
            
            ipx, ipy = int(px_n), int(py_n)
            
            if grid_exist[ipx, ipy] == 1:
                mat = grid_material[ipx, ipy]
                cl_state = grid_count_cl[ipx, ipy]
                nx, ny = get_surface_normal(ipx, ipy)
                
                # --- 入射角判断 ---
                cos_theta = -(vx * nx + vy * ny)
                if cos_theta < 0: cos_theta = 0
                
                if is_ion == 0:
                    # >>> 中性粒子：产物是固体 $SiCl_x(s)$ <<<
                    # 规则：概率是吸附率，绝不删点，只加状态
                    prob = 0.0
                    if mat == 1: # Si
                        if cl_state == 0: prob = 0.6
                        elif cl_state == 1: prob = 0.3
                        elif cl_state == 2: prob = 0.1
                    elif mat == 2: # Mask
                        prob = 0.01 
                    
                    if ti.random() < prob:
                        if grid_count_cl[ipx, ipy] < 3:
                            grid_count_cl[ipx, ipy] += 1
                        alive = False # 被吸附
                    else:
                        # 反射
                        vx, vy = get_reflection_vector(vx, vy, nx, ny, 0)
                        px, py = px_n + nx, py_n + ny
                
                else:
                    # >>> 离子：产物是气体 $SiCl_x(g)$ <<<
                    # 规则：概率是刻蚀产率，直接删点变真空
                    yield_prob = 0.0
                    f_alpha = 1.0 if cos_theta > 0.707 else cos_theta/0.707
                    
                    if mat == 1: # Si
                        if cl_state == 0: yield_prob = 0.05 * f_alpha
                        elif cl_state == 1: yield_prob = 0.3 * f_alpha
                        elif cl_state == 2: yield_prob = 0.7 * f_alpha
                        elif cl_state == 3: yield_prob = 0.95 * f_alpha
                    elif mat == 2: # Mask
                        yield_prob = 0.02 * f_alpha
                    
                    if ti.random() < yield_prob:
                        # 生成气体，删除原子！
                        grid_exist[ipx, ipy] = 0
                        grid_material[ipx, ipy] = 0
                        grid_count_cl[ipx, ipy] = 0
                        alive = False
                    else:
                        # 离子反射
                        vx, vy = get_reflection_vector(vx, vy, nx, ny, 1)
                        px, py = px_n + nx, py_n + ny
            else:
                px, py = px_n, py_n

# ================= 4. 主程序 =================

def main():
    init_grid()
    cmap = ListedColormap(["#008CFF", '#00008B', "#00FFFF"]) # 真空, Si, Mask
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    for i in range(num_batches):
        simulate_batch()
        if i % 25 == 0:
            ti.sync()
            mat_data = grid_material.to_numpy()
            ax.clear()
            ax.imshow(mat_data.T, cmap=cmap, vmin=0, vmax=2, origin='upper')
            ax.set_title(f"Particles: {i*BATCH_SIZE} | Logic: Gas=Delete, Solid=Keep")
            plt.pause(0.01)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()