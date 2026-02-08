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
NUM_TRACE = 100       # 追踪粒子数量
MAX_STEPS = 500       # 最大步数
RATIO = 0.5           # 离子/中性比例

# --- 【新增】反射概率参数 ---
# 离子反射阈值角 (例如 45 度)
ION_TH_ANGLE_DEG = 45.0
ION_TH_ANGLE_RAD = ION_TH_ANGLE_DEG * math.pi / 180.0

# 中性粒子反射概率 (1.0 = 永远反射, 0.0 = 永远吸收)
NEUTRAL_REFLECT_PROB = 0.8 

# --- 数据场 ---
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))

# --- 轨迹存储器 ---
path_x = ti.field(dtype=ti.f32, shape=(NUM_TRACE, MAX_STEPS))
path_y = ti.field(dtype=ti.f32, shape=(NUM_TRACE, MAX_STEPS))
path_len = ti.field(dtype=ti.i32, shape=(NUM_TRACE))
particle_type = ti.field(dtype=ti.i32, shape=(NUM_TRACE)) # 0=中性, 1=离子

# ================= 2. 辅助物理函数 =================

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
    """ 计算反射后的速度向量 """
    rvx, rvy = 0.0, 0.0
    
    if is_ion == 1:
        # === 离子：镜面反射 ===
        dot = vx * nx + vy * ny
        rvx = vx - 2 * dot * nx
        rvy = vy - 2 * dot * ny
    else:
        # === 中性粒子：漫反射 (Cosine Law) ===
        tx, ty = -ny, nx
        sin_theta = (ti.random() - 0.5) * 2.0 
        cos_theta = ti.sqrt(1.0 - sin_theta**2)
        rvx = nx * cos_theta + tx * sin_theta
        rvy = ny * cos_theta + ty * sin_theta

    return rvx, rvy

# ================= 3. 几何初始化 =================
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

# ================= 4. 带概率计算的追踪核心 =================

@ti.kernel
def trace_particles_with_prob():
    for k in range(NUM_TRACE):
        # --- A. 生成 ---
        # 让粒子分布在整个区域上方
        px, py = ti.random() * (ROWS - 1), 1.0
        # 或者固定位置调试: px, py = 200.0 + 5*k, 1.0 
        
        is_ion = 0
        if ti.random() > RATIO: is_ion = 1 
        particle_type[k] = is_ion 
        
        # 初始角度：离子较直，中性较散
        sigma = (5.0 if is_ion==1 else 45.0) * (math.pi/180)
        angle = ti.randn() * sigma
        # 限制角度防止横着飞
        angle = max(min(angle, math.pi/2 - 0.1), -math.pi/2 + 0.1)
        
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive = True
        steps = 0
        
        # --- B. 飞行循环 ---
        while alive and steps < MAX_STEPS:
            # 记录轨迹
            path_x[k, steps] = px
            path_y[k, steps] = py
            steps += 1
            
            # 试探步
            px_n, py_n = px + vx * 1.0, py + vy * 1.0
            
            # 边界处理
            if px_n < 0: px_n += ROWS
            if px_n >= ROWS: px_n -= ROWS
            if py_n < 0 or py_n >= COLS: 
                alive = False
                break
            
            ipx, ipy = int(px_n), int(py_n)
            
            # === 碰撞检测 ===
            if grid_exist[ipx, ipy] > 0.5:
                # 1. 计算法线
                nx, ny = get_surface_normal(ipx, ipy)
                
                # 2. 计算入射角 (Incident Angle)
                # 速度 v 与 法线 n 是迎头撞击，夹角 > 90度。
                # 我们需要的是入射向量反向与法线的夹角，或者直接用点积公式
                # dot < 0 代表迎头撞击
                dot = vx * nx + vy * ny 
                
                # 入射角 alpha (0 = 垂直撞击, pi/2 = 擦边)
                # cos(alpha) = |dot| / (|v|*|n|) -> |dot| (因归一化了)
                cos_alpha = ti.abs(dot)
                # 限制范围防止 NaN
                cos_alpha = min(1.0, max(0.0, cos_alpha))
                alpha_rad = ti.acos(cos_alpha)
                
                # 3. 【新增】计算反射概率 P_reflect
                p_reflect = 0.0
                
                if is_ion == 1:
                    # === 离子反射概率逻辑 ===
                    # 阈值：ION_TH_ANGLE_RAD
                    # 0 到 阈值 -> p=0
                    # 阈值 到 pi/2 -> p 线性从 0 到 1
                    
                    if alpha_rad < ION_TH_ANGLE_RAD:
                        p_reflect = 0.0 # 垂直撞击，直接吸收/刻蚀
                    else:
                        # 线性插值
                        # 进度 = (当前角 - 阈值) / (90度 - 阈值)
                        interval = (math.pi / 2.0) - ION_TH_ANGLE_RAD
                        if interval > 1e-6:
                            p_reflect = (alpha_rad - ION_TH_ANGLE_RAD) / interval
                        else:
                            p_reflect = 1.0
                            
                    # 钳制概率在 0-1 之间
                    p_reflect = min(1.0, max(0.0, p_reflect))
                    
                else:
                    # === 中性粒子反射概率逻辑 ===
                    # 随机 (即常数概率)
                    p_reflect = NEUTRAL_REFLECT_PROB
                
                # 4. 掷骰子决定命运
                if ti.random() < p_reflect:
                    # >> 反射 <<
                    vx, vy = get_reflection_vector(vx, vy, nx, ny, is_ion)
                    # 推离墙壁
                    px = px + nx * 1.5
                    py = py + ny * 1.5
                else:
                    # >> 吸收/反应/消失 <<
                    alive = False
                    # 此时 steps 循环结束，粒子停在墙上
            else:
                # 没撞到，更新位置
                px, py = px_n, py_n
        
        path_len[k] = steps

# ================= 5. 绘图 =================

def main():
    print(">>> 初始化几何...")
    init_grid()
    
    print(f">>> 追踪 {NUM_TRACE} 个粒子...")
    print(f"    - 离子反射阈值: {ION_TH_ANGLE_DEG}°")
    print(f"    - 中性粒子反射率: {NEUTRAL_REFLECT_PROB}")
    
    trace_particles_with_prob()
    ti.sync()
    
    # 提取数据
    p_x = path_x.to_numpy()
    p_y = path_y.to_numpy()
    p_l = path_len.to_numpy()
    p_t = particle_type.to_numpy()
    exist = grid_exist.to_numpy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 画背景结构
    ax.imshow(exist.T, cmap='Greys', alpha=0.3, origin='upper', extent=[0, ROWS, COLS, 0])
    
    print(">>> 正在绘制...")
    ion_cnt = 0
    neu_cnt = 0
    
    for k in range(NUM_TRACE):
        length = p_l[k]
        if length < 2: continue 
        
        xs = p_x[k, :length]
        ys = p_y[k, :length]
        
        # 离子 = 红, 中性 = 绿
        if p_t[k] == 1:
            ax.plot(xs, ys, '-', color='red', alpha=0.6, linewidth=1.0)
            # 在终点画个叉，表示被吸收
            if length < MAX_STEPS: 
                ax.plot(xs[-1], ys[-1], 'x', color='red', markersize=5)
            ion_cnt += 1
        else:
            ax.plot(xs, ys, '--', color='green', alpha=0.4, linewidth=0.8)
            neu_cnt += 1

    ax.set_xlim(0, ROWS)
    ax.set_ylim(deep_border + 300, 0)
    ax.set_title(f"Reflection Probability Test\nIon Thresh={ION_TH_ANGLE_DEG}deg, Neutral Prob={NEUTRAL_REFLECT_PROB}")
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label=f'Ion ({ion_cnt})'),
        Line2D([0], [0], color='green', lw=2, linestyle='--', label=f'Neutral ({neu_cnt})'),
        Line2D([0], [0], color='black', marker='x', lw=0, label='Absorbed/Etched')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.show()

if __name__ == "__main__":
    main()