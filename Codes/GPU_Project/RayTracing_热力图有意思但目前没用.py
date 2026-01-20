import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# ================= 1. 环境与常量 =================
ti.init(arch=ti.gpu)

ROWS, COLS = 1000, 700
vacuum = 100
deep_border = 230
left_border = 150
right_border = 250
Space = 100
Num = 3
CD = right_border - left_border

# 测试用的粒子数量 (不需要跑几百万，几万个就够看分布了)
TEST_PARTICLES = 500000 
BATCH_SIZE = 2000
RATIO = 10.0 / 11.0  # 控制角度分布 (离子/中性)

# --- 数据场 ---
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS)) # 0=Vac, 1=Si, 2=Mask

# 【新增】专门用于记录撞击位置的计数器
hit_map = ti.field(dtype=ti.i32, shape=(ROWS, COLS))

# ================= 2. 几何初始化 (保持原样) =================
@ti.kernel
def init_grid():
    angle_rad = 5 * math.pi / 180
    k_mask = ti.abs(ti.tan(angle_rad))
    for i, j in grid_exist:
        if j <= vacuum:
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
                        grid_material[x, y] = 2; grid_exist[x, y] = 1.0
                
                for x in range(left_side, right_side):
                    if  left_current < x < right_current:
                        grid_exist[x, y] = 0.0; grid_material[x, y] = 0
                    else:
                        grid_exist[x, y] = 1.0; grid_material[x, y] = 2 
        
        last_right_side = current_right_border + int(Space / 2)
        if last_right_side <= i < ROWS - 1 and vacuum < j < deep_border:
             grid_exist[i, j] = 1.0; grid_material[i, j] = 2 

        if j < COLS:
            if grid_exist[i, j] == 0.0 and j >= deep_border:
                 grid_exist[i, j] = 1.0; grid_material[i, j] = 1

# ================= 3. 核心测试模块：Ray Tracing =================

@ti.kernel
def test_ray_tracing():
    """
    只负责：生成 -> 飞行 -> 记录撞击点
    完全剥离：反射、刻蚀、物理溅射
    """
    for k in range(BATCH_SIZE):
        # --- A. 粒子生成 (完全复刻 simulate_batch) ---
        px, py = ti.random() * (ROWS - 1), 1.0
        
        # 决定是离子还是中性 (决定角度分布)
        is_ion = 0
        if ti.random() > RATIO: is_ion = 1 
        
        sigma = (1.91 if is_ion==1 else 7.64) * (math.pi/180)
        angle = ti.randn() * sigma
        angle = max(min(angle, math.pi/2), -math.pi/2)
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive, steps = True, 0
        
        # --- B. 射线追踪循环 ---
        while alive and steps < 2000:
            steps += 1
            # 1. 移动
            px_n, py_n = px + vx * 1.1, py + vy * 1.1
            
            # 2. 周期性边界
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            
            # 3. 出界判定 (打到底部或者飞出顶部)
            if py_n < 0 or py_n >= COLS: 
                alive = False
                break
                
            ipx, ipy = int(px_n), int(py_n)
            
            # 4. 碰撞检测 (这是 Ray Tracing 的终点)
            if grid_exist[ipx, ipy] > 0.5:
                # === 记录撞击 ===
                # 在 hit_map 上+1，稍后画出来看
                hit_map[ipx, ipy] += 1
                
                # === 停止 ===
                # 在这个测试里，撞到就停，不反弹
                alive = False 
            else:
                # 没撞到，继续飞
                px, py = px_n, py_n

# ================= 4. 主程序与可视化 =================

def main():
    print(">>> 正在初始化几何结构...")
    init_grid()
    
    # 清空计数器
    hit_map.fill(0)
    
    print(f">>> 开始发射测试粒子: {TEST_PARTICLES} 个")
    num_batches = TEST_PARTICLES // BATCH_SIZE
    
    for i in range(num_batches):
        test_ray_tracing()
        if i % 5 == 0:
            print(f"进度: {i/num_batches:.1%}", end='\r')
    
    ti.sync()
    print("\n>>> 计算完成，正在绘图...")

    # --- 获取数据 ---
    hits = hit_map.to_numpy()
    exist = grid_exist.to_numpy()

    # --- 绘图验证 ---
    plt.figure(figsize=(12, 8))
    
    # 1. 画出几何轮廓 (灰色背景)
    # 用简单的二值图显示结构
    plt.imshow(exist.T, cmap='Greys', alpha=0.3, origin='upper')
    
    # 2. 画出撞击热力图 (红黄色)
    # 使用 masked array，只显示有撞击的地方
    hits_masked = np.ma.masked_where(hits == 0, hits)
    plt.imshow(hits_masked.T, cmap='jet', interpolation='nearest', origin='upper', alpha=0.9)
    
    plt.colorbar(label='Hit Count (Flux Intensity)')
    plt.title(f"Ray Tracing Test (Incident Angle Check)\nParticles: {TEST_PARTICLES}, Ratio(N/Total): {RATIO:.2f}")
    plt.xlabel("X (Width)")
    plt.ylabel("Y (Depth)")
    
    # 限制显示区域，方便看清楚沟槽
    plt.xlim(0, ROWS)
    plt.ylim(deep_border + 100, 0) # 只看上半部分结构
    
    print(">>> 图表已生成。")
    print(">>> 观察重点：")
    print("   1. 顶部平面应该是红色的（撞击最多）。")
    print("   2. 沟槽侧壁应该有渐变（上多下少）。")
    print("   3. 沟槽底部角落应该几乎没有撞击（蓝色或透明），这就是Shadowing Effect。")
    plt.show()

if __name__ == "__main__":
    main()