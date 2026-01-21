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
NUM_TRACE = 50        # 追踪 100 个粒子
MAX_STEPS = 300       # 步数加多一点，让它们多弹一会儿
RATIO = 0  # 离子/中性比例

# --- 数据场 ---
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))

# --- 轨迹存储器 ---
path_x = ti.field(dtype=ti.f32, shape=(NUM_TRACE, MAX_STEPS))
path_y = ti.field(dtype=ti.f32, shape=(NUM_TRACE, MAX_STEPS))
path_len = ti.field(dtype=ti.i32, shape=(NUM_TRACE))
particle_type = ti.field(dtype=ti.i32, shape=(NUM_TRACE)) # 0=中性, 1=离子

# ================= 2. 辅助物理函数 (法线 & 反射) =================

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
    """ 
    计算反射后的速度向量 
    离子 = 镜面反射
    中性 = 漫反射 (Lambertian)
    """
    rvx, rvy = 0.0, 0.0
    
    if is_ion == 1:
        # === 离子：镜面反射 ===
        # v_out = v_in - 2 * (v_in . n) * n
        dot = vx * nx + vy * ny
        rvx = vx - 2 * dot * nx
        rvy = vy - 2 * dot * ny
    else:
        # === 中性粒子：漫反射 ===
        # 1. 构造切向向量
        tx, ty = -ny, nx
        
        # 2. 随机生成角度 (Lambertian Cosine Law)
        # sin_theta 均匀分布
        sin_theta = (ti.random() - 0.5) * 2.0 
        cos_theta = ti.sqrt(1.0 - sin_theta**2)
        
        # 3. 旋转法线
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

# ================= 4. 带反射的追踪核心 =================

@ti.kernel
def trace_particles_with_bounce():
    for k in range(NUM_TRACE):
        # --- A. 生成 ---
        # px, py = ti.random() * (ROWS - 1), 1.0
        px, py = 200.0 + 10*k, 1.0
        
        is_ion = 0
        if ti.random() > RATIO: is_ion = 1 
        particle_type[k] = is_ion 
        
        # sigma = (1.91 if is_ion==1 else 7.64) * (math.pi/180)
        # angle = ti.randn() * sigma
        # angle = max(min(angle, math.pi/2), -math.pi/2)
        angle = math.pi/10
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive = True
        steps = 0
        
        # --- B. 飞行、碰撞、反弹 ---
        while alive and steps < MAX_STEPS:
            # 记录轨迹
            path_x[k, steps] = px
            path_y[k, steps] = py
            steps += 1
            
            # 试探步 (Look-ahead)
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
                # 1. 碰到墙了，计算法线
                nx, ny = get_surface_normal(ipx, ipy)
                
                # 2. 计算反射后的新速度
                vx, vy = get_reflection_vector(vx, vy, nx, ny, is_ion)
                
                # 3. 【防卡死技巧】
                # 不要让粒子停在墙里，也不要只改变速度
                # 把它沿着法线方向推离墙壁一点点，确保下一帧它在真空中
                px = px + nx * 1.5 
                py = py + ny * 1.5
                
                # 这里不更新 path，让它在下一帧循环时记录新位置，形成折线
            else:
                # 没撞到，正常更新位置
                px, py = px_n, py_n
        
        path_len[k] = steps

# ================= 5. 绘图 =================

def main():
    print(">>> 初始化几何...")
    init_grid()
    
    print(f">>> 正在追踪 {NUM_TRACE} 个粒子 (含反射机制)...")
    trace_particles_with_bounce()
    ti.sync()
    
    p_x = path_x.to_numpy()
    p_y = path_y.to_numpy()
    p_l = path_len.to_numpy()
    p_t = particle_type.to_numpy()
    exist = grid_exist.to_numpy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 画背景
    ax.imshow(exist.T, cmap='Greys', alpha=0.3, origin='upper', extent=[0, ROWS, COLS, 0])
    
    print(">>> 正在绘制弹跳轨迹...")
    ion_cnt = 0
    neu_cnt = 0
    
    for k in range(NUM_TRACE):
        length = p_l[k]
        if length < 5: continue 
        
        xs = p_x[k, :length]
        ys = p_y[k, :length]
        
        if p_t[k] == 1:
            # 离子：红色实线
            ax.plot(xs, ys, '-', color='red', alpha=0.5, linewidth=0.8)
            ion_cnt += 1
        else:
            # 中性：绿色虚线
            ax.plot(xs, ys, '--', color='green', alpha=0.4, linewidth=0.8)
            neu_cnt += 1

    ax.set_xlim(0, ROWS)
    ax.set_ylim(deep_border + 200, 0)
    ax.set_title(f"Particle Reflection Test\nRed=Ion (Specular), Green=Neutral (Diffuse)\nTotal: {NUM_TRACE}")
    
    # 标注说明
    plt.text(10, 20, f"Simulated Steps: {MAX_STEPS}", color='blue')
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', lw=2, label=f'Ion ({ion_cnt})'),
                       Line2D([0], [0], color='green', lw=2, linestyle='--', label=f'Neutral ({neu_cnt})')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    print(">>> 绘图完成。请观察绿色的乱跳和红色的规则反弹。")
    plt.show()

if __name__ == "__main__":
    main()