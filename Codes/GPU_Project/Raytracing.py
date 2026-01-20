import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import math

# ================= 1. 环境初始化 =================
ti.init(arch=ti.gpu)

ROWS, COLS = 1000, 700
vacuum = 100
deep_border = 230
left_border = 150
right_border = 250
Space = 100
Num = 3
CD = right_border - left_border

# --- 轨迹追踪专用参数 ---
NUM_TRACE = 100        # 只追踪 100 个粒子，不然图会看不清
MAX_STEPS = 2000       # 每个粒子最多飞多少步
RATIO = 10.0 / 11.0  # 离子/中性比例

# --- 数据场 ---
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))

# --- 轨迹存储器 (用于保存路径) ---
# shape = (粒子编号, 步数)
path_x = ti.field(dtype=ti.f32, shape=(NUM_TRACE, MAX_STEPS))
path_y = ti.field(dtype=ti.f32, shape=(NUM_TRACE, MAX_STEPS))
path_len = ti.field(dtype=ti.i32, shape=(NUM_TRACE)) # 记录每个粒子实际走了多少步
particle_type = ti.field(dtype=ti.i32, shape=(NUM_TRACE)) # 0=中性, 1=离子

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

# ================= 3. 轨迹追踪核心 =================

@ti.kernel
def trace_particles():
    # 对每一个追踪粒子并行计算
    for k in range(NUM_TRACE):
        # --- A. 生成 ---
        # px, py = ti.random() * (ROWS - 1), 1.0
        px, py = 200.0, 1.0
        
        is_ion = 0
        if ti.random() > RATIO: is_ion = 1 
        particle_type[k] = is_ion # 记录类型以便画图
        
        sigma = (1.91 if is_ion==1 else 7.64) * (math.pi/180)
        angle = ti.randn() * sigma
        angle = max(min(angle, math.pi/2), -math.pi/2)
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive = True
        steps = 0
        
        # --- B. 飞行与记录 ---
        while alive and steps < MAX_STEPS:
            # 1. 【记录当前位置】
            path_x[k, steps] = px
            path_y[k, steps] = py
            steps += 1
            
            # 2. 移动
            px_n, py_n = px + vx * 1.1, py + vy * 1.1
            
            # 3. 边界处理
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            if py_n < 0 or py_n >= COLS: 
                alive = False
                break
            
            ipx, ipy = int(px_n), int(py_n)
            
            # 4. 碰撞检测
            if grid_exist[ipx, ipy] > 0.5:
                # 记录最后一步撞击的位置
                path_x[k, steps] = px_n
                path_y[k, steps] = py_n
                steps += 1
                alive = False # 撞墙即停
            else:
                px, py = px_n, py_n
        
        path_len[k] = steps

# ================= 4. 绘图程序 =================

def main():
    print(">>> 初始化几何...")
    init_grid()
    
    print(f">>> 正在追踪 {NUM_TRACE} 个粒子的轨迹...")
    trace_particles()
    ti.sync()
    
    # 获取数据回 CPU
    p_x = path_x.to_numpy()
    p_y = path_y.to_numpy()
    p_l = path_len.to_numpy()
    p_t = particle_type.to_numpy()
    exist = grid_exist.to_numpy()
    
    # --- 开始画图 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. 画背景结构 (灰色)
    ax.imshow(exist.T, cmap='Greys', alpha=0.3, origin='upper', extent=[0, ROWS, COLS, 0])
    
    print(">>> 正在绘制线条...")
    ion_count = 0
    neutral_count = 0
    
    for k in range(NUM_TRACE):
        # 提取当前粒子的有效路径
        length = p_l[k]
        if length < 2: continue # 没飞起来的不画
        
        xs = p_x[k, :length]
        ys = p_y[k, :length]
        
        # 设定颜色和线型
        if p_t[k] == 1:
            # 离子：红色，细实线，透明度高一点
            color = 'red'
            style = '-'
            alpha = 0.6
            lw = 1.0
            ion_count += 1
        else:
            # 中性粒子：绿色，虚线，透明度低一点
            color = 'green' 
            style = '--'
            alpha = 0.4
            lw = 0.8
            neutral_count += 1
            
        ax.plot(xs, ys, linestyle=style, color=color, alpha=alpha, linewidth=lw)
        
        # 标出终点 (撞击点)
        ax.scatter([xs[-1]], [ys[-1]], s=10, c=color, marker='x')

    ax.set_xlim(0, ROWS)
    ax.set_ylim(deep_border + 100, 0)
    ax.set_title(f"Particle Trajectories (Total: {NUM_TRACE})\nRed=Ion (Narrow Angle), Green=Neutral (Wide Angle)")
    
    # 手动加个图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', lw=2, label=f'Ion ({ion_count})'),
                       Line2D([0], [0], color='green', lw=2, linestyle='--', label=f'Neutral ({neutral_count})')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    print(">>> 绘图完成。")
    plt.show()

if __name__ == "__main__":
    main()