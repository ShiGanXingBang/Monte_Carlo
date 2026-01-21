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

# --- 模拟参数 ---
NUM_TRACE = 100         # 追踪 100 个粒子
MAX_STEPS = 5000        # 给它们足够的时间去反弹
Prob_Etch = 0.1         # 10% 概率发生刻蚀 (反应并消失)
Prob_Adsorb = 0.2       # 20% 概率被吸附 (粘住并消失)
# 剩余 70% 概率发生反射

# --- 数据场 ---
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))

# --- 轨迹与命运记录 ---
path_x = ti.field(dtype=ti.f32, shape=(NUM_TRACE, MAX_STEPS))
path_y = ti.field(dtype=ti.f32, shape=(NUM_TRACE, MAX_STEPS))
path_len = ti.field(dtype=ti.i32, shape=(NUM_TRACE))
# 命运标记: 0=飞出界, 1=吸附(Adsorbed), 2=刻蚀(Etched)
particle_fate = ti.field(dtype=ti.i32, shape=(NUM_TRACE)) 

# ================= 2. 物理函数 (核心) =================

@ti.func
def get_surface_normal(px: int, py: int):
    """ 计算表面法线 """
    nx, ny = 0.0, 0.0
    for i, j in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        if 0 <= px + i < ROWS and 0 <= py + j < COLS:
            if grid_exist[px + i, py + j] < 0.5:
                nx += float(i)
                ny += float(j)
    norm = ti.sqrt(nx**2 + ny**2) + 1e-6
    return nx / norm, ny / norm

@ti.func
def get_lambertian_reflection(nx: float, ny: float):
    """ 
    【中性粒子专用】Lambertian 漫反射 
    返回一个新的随机反射方向 (vx, vy)
    """
    # 1. 构造切向向量
    tx, ty = -ny, nx
    
    # 2. 随机生成角度 (符合余弦分布)
    sin_theta = (ti.random() - 0.5) * 2.0 
    cos_theta = ti.sqrt(1.0 - sin_theta**2)
    
    # 3. 向量合成
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

# ================= 4. 中性粒子模拟流程 =================

@ti.kernel
def simulate_neutrals():
    for k in range(NUM_TRACE):
        # --- 1. 生成 (强制为中性粒子，角度较宽) ---
        px, py = ti.random() * (ROWS - 50), 1.0
        
        # 中性粒子角度分布 (sigma大)
        sigma = 7.64 * (math.pi/180) 
        angle = ti.randn() * sigma
        angle = max(min(angle, math.pi/2), -math.pi/2)
        vx, vy = ti.sin(angle), ti.cos(angle)
        
        alive = True
        steps = 0
        fate = 0 # 默认命运：飞出界
        
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
                alive = False # 飞出界了
                break
            
            ipx, ipy = int(px_n), int(py_n)
            
            # --- 2. 碰撞逻辑 ---
            if grid_exist[ipx, ipy] > 0.5:
                # 记录最后一步位置
                path_x[k, steps] = px_n
                path_y[k, steps] = py_n
                steps += 1
                
                # >> 命运轮盘赌 <<
                rng = ti.random()
                
                if rng < Prob_Etch:
                    # [结局 A] 发生刻蚀反应
                    fate = 2
                    alive = False # 粒子消耗掉了
                    
                elif rng < Prob_Etch + Prob_Adsorb:
                    # [结局 B] 被吸附 (Stick)
                    fate = 1
                    alive = False # 粒子粘在墙上了
                    
                else:
                    # [结局 C] 漫反射 (Bounce)
                    nx, ny = get_surface_normal(ipx, ipy)
                    
                    # 漫反射计算新速度
                    vx, vy = get_lambertian_reflection(nx, ny)
                    
                    # 推离墙壁，继续存活
                    px = px + nx * 1.5
                    py = py + ny * 1.5
                    
                    # 注意：这里活着，alive 依然是 True，循环继续
            else:
                # 没撞到，继续飞
                px, py = px_n, py_n
        
        path_len[k] = steps
        particle_fate[k] = fate

# ================= 5. 绘图与分析 =================

def main():
    init_grid()
    print(">>> 正在模拟中性粒子的漫反射与反应...")
    simulate_neutrals()
    ti.sync()
    
    # 提取数据
    p_x = path_x.to_numpy()
    p_y = path_y.to_numpy()
    p_l = path_len.to_numpy()
    p_f = particle_fate.to_numpy()
    exist = grid_exist.to_numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. 画背景
    ax.imshow(exist.T, cmap='Greys', alpha=0.3, origin='upper', extent=[0, ROWS, COLS, 0])
    
    print(">>> 正在绘制轨迹...")
    
    legend_added = set()
    
    for k in range(NUM_TRACE):
        length = p_l[k]
        if length < 2: continue
        
        xs = p_x[k, :length]
        ys = p_y[k, :length]
        
        # 画轨迹线 (绿色虚线)
        ax.plot(xs, ys, '--', color='green', alpha=0.3, linewidth=0.8)
        
        # 画终点标记
        last_x, last_y = xs[-1], ys[-1]
        fate = p_f[k]
        
        if fate == 2: # Etch
            ax.scatter([last_x], [last_y], c='red', marker='*', s=100, zorder=10, 
                       label='Etched (Reaction)' if 'Etch' not in legend_added else "")
            legend_added.add('Etch')
            
        elif fate == 1: # Adsorb
            ax.scatter([last_x], [last_y], c='blue', marker='o', s=40, zorder=10, 
                       label='Adsorbed (Stuck)' if 'Ads' not in legend_added else "")
            legend_added.add('Ads')
            
        # fate == 0 是飞出界，不标记
        
    ax.set_xlim(0, ROWS)
    ax.set_ylim(deep_border + 200, 0)
    ax.set_title("Neutral Particle Demo: Diffuse Reflection & Reaction\nProbabilities: Etch 10% | Adsorb 20% | Reflect 70%")
    
    ax.legend(loc='upper right')
    print(">>> 绘图完成。")
    print("   * 绿色虚线：粒子飞行路径（注意观察它们像没头苍蝇一样乱撞）")
    print("   * 红色五角星：发生刻蚀的位置（这里会被挖掉）")
    print("   * 蓝色圆点：被吸附的位置（这里Cl浓度会增加）")
    plt.show()

if __name__ == "__main__":
    main()