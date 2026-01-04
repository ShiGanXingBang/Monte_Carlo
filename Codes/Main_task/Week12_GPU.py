import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# 1. 初始化 Taichi (使用 GPU)
ti.init(arch=ti.gpu)

# 2. 常量定义
ROWS, COLS = 800, 700
TOTAL_PARTICLES = 1000000
BATCH_SIZE = 5000  # 每次并行发射 5000 个粒子
RATIO = (10/11)    # 离子/中性粒子比例

# 3. 定义数据场 (存储网格信息)
grid_exist = ti.field(dtype=ti.i32, shape=(ROWS, COLS))      # 1: 存在, 0: 真空
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   # Cl 吸附数 (0-4)
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   # 1: Si, 2: Hardmask

# 4. 物理逻辑函数
@ti.func
def get_surface_normal(px, py):
    """
    通过扫描邻域确定表面法线 (GPU版)
    寻找周围 5x5 区域内空位(真空)最集中的方向
    """
    nx, ny = 0.0, 0.0
    for i, j in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        if 0 <= px + i < ROWS and 0 <= py + j < COLS:
            if grid_exist[px + i, py + j] == 0:
                nx += float(i)
                ny += float(j)
    norm = ti.sqrt(nx**2 + ny**2) + 1e-6
    return nx / norm, ny / norm

@ti.kernel
def init_grid():
    """初始化几何结构 (沟槽形状)"""
    k_mask = ti.abs(ti.tan(5 * math.pi / 90))
    for i, j in grid_exist:
        if j < 100: # 真空层
            grid_exist[i, j] = 0
        elif j < 200: # 遮罩层深度
            offset = int((200 - j) * k_mask)
            if (200 - offset) < i < (600 + offset):
                grid_exist[i, j] = 0
            else:
                grid_exist[i, j] = 1
                grid_material[i, j] = 2 # Hardmask
        else:
            grid_exist[i, j] = 1
            grid_material[i, j] = 1 # Si

@ti.kernel
def simulate_batch(seed: int):
    """并行模拟粒子行为"""
    for k in range(BATCH_SIZE):
        # ti.math.set_rng_pos(seed * BATCH_SIZE + k)
        
        # 发射参数
        px = ti.random() * (ROWS - 1)
        py = 1.0
        species = 1 if ti.random() > RATIO else 0 # 1:离子, 0:中性
        
        # 入射角 (高斯分布)
        sigma = 1.91 * (math.pi/180) if species == 1 else 7.64 * (math.pi/180)
        angle = ti.randn() * sigma
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive = True
        steps = 0
        while alive and steps < 3000:
            steps += 1
            # 步进
            px_new = px + vx
            py_new = py + vy
            
            # X轴环绕
            if px_new < 0: px_new += ROWS
            if px_new >= ROWS: px_new -= ROWS
            
            # Y轴边界检查
            if py_new < 0 or py_new >= COLS:
                alive = False
                break
                
            ipx, ipy = int(px_new), int(py_new)
            
            # 碰撞检测
            if grid_exist[ipx, ipy] == 1:
                # 命中物质
                mat = grid_material[ipx, ipy]
                cl_n = grid_count_cl[ipx, ipy]
                
                # --- 核心刻蚀逻辑 ---
                should_etch = False
                if species == 1: # 离子
                    if mat == 1 and ti.random() < 0.1: # 物理溅射
                        should_etch = True
                    else: # 化学辅助
                        prob = 0.3 # 离子在Si上的基础反应率
                        if mat == 2: prob *= 0.02 # Mask 较难刻蚀
                        if ti.random() < prob: should_etch = True
                else: # 中性粒子
                    # 反应概率根据 Cl 吸附量变化 (0, 0.1, 0.2, 0.3, 1.0)
                    prob = 0.0
                    if cl_n == 1: prob = 0.1
                    elif cl_n == 2: prob = 0.2
                    elif cl_n == 3: prob = 0.3
                    elif cl_n == 4: prob = 1.0
                    
                    if mat == 2: prob *= 0.1
                    if ti.random() < prob: should_etch = True
                
                if should_etch:
                    grid_exist[ipx, ipy] = 0
                    alive = False
                else:
                    # --- 反射逻辑 ---
                    nx, ny = get_surface_normal(ipx, ipy)
                    cos_theta = -(vx * nx + vy * ny) # 入射角余弦
                    
                    # 简单反射判定
                    if ti.random() < 0.5: # 假设反射概率
                        # 镜面反射：v_out = v_in - 2(v_in·n)n
                        dot = vx*nx + vy*ny
                        vx = vx - 2 * dot * nx
                        vy = vy - 2 * dot * ny
                        px, py = px_new, py_new # 更新位置
                    else:
                        # 粒子被捕获或吸附
                        if species == 0 and grid_count_cl[ipx, ipy] < 4:
                            grid_count_cl[ipx, ipy] += 1
                        alive = False
            else:
                # 在真空中飞行
                px, py = px_new, py_new

# 5. 执行模拟
init_grid()
print("GPU 模拟启动...")
start = time.time()
for i in range(TOTAL_PARTICLES // BATCH_SIZE):
    simulate_batch(i)
ti.sync()
print(f"计算完成，耗时: {time.time() - start:.2f} s")

# 6. 显示结果
res = grid_exist.to_numpy()
plt.imshow(res.T, cmap='gray')
plt.title("GPU Accelerated Monte Carlo Result")
plt.show()