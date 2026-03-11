import taichi as ti
import numpy as np
import math
import time

# ================= 1. 环境初始化 =================
# 启用 GPU 并使用 vulkan 或 cuda 作为 3D 渲染后端
ti.init(arch=ti.gpu)

# --- 3D 全局常量 ---
# 为了保证实时 3D 渲染帧率，我们将网格调整为 100x100x200
SIZE_X = 100
SIZE_Y = 100
SIZE_Z = 200 

vacuum = 30
deep_border = 60
HOLE_RADIUS = 20 # 掩膜圆孔半径

TOTAL_PARTICLES = 20000000
BATCH_SIZE = 8000
RATIO = 20.0 / 21.0  # 中性/总粒子比例

# --- Taichi 数据场 (3D) ---
grid_exist = ti.field(dtype=ti.f32, shape=(SIZE_X, SIZE_Y, SIZE_Z))      
grid_material = ti.field(dtype=ti.i32, shape=(SIZE_X, SIZE_Y, SIZE_Z))   
grid_count_cl = ti.field(dtype=ti.i32, shape=(SIZE_X, SIZE_Y, SIZE_Z))   

# --- 3D 渲染专用数据场 ---
MAX_RENDER_PARTICLES = SIZE_X * SIZE_Y * SIZE_Z
render_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RENDER_PARTICLES)
render_color = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RENDER_PARTICLES)
particle_cnt = ti.field(dtype=ti.i32, shape=())

# ================= 2. 物理辅助函数 (3D化) =================

@ti.func
def get_surface_normal_3d(px: int, py: int, pz: int):
    """ 计算 3D 表面法线 (指向真空) """
    nx, ny, nz = 0.0, 0.0, 0.0
    for i, j, k in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
        if 0 <= px + i < SIZE_X and 0 <= py + j < SIZE_Y and 0 <= pz + k < SIZE_Z:
            if grid_exist[px + i, py + j, pz + k] < 0.5:
                nx += float(i)
                ny += float(j)
                nz += float(k)
    norm = ti.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
    return nx / norm, ny / norm, nz / norm

@ti.func
def get_reflection_vector_3d(vx: float, vy: float, vz: float, nx: float, ny: float, nz: float, is_ion: int):
    """ 计算 3D 反射向量 """
    rvx, rvy, rvz = 0.0, 0.0, 0.0
    if is_ion == 1:
        # 3D 镜面反射: v - 2(v.n)n
        dot = vx * nx + vy * ny + vz * nz
        rvx = vx - 2 * dot * nx
        rvy = vy - 2 * dot * ny
        rvz = vz - 2 * dot * nz
    else:
        # 3D Lambertian 漫反射
        rx = ti.randn(); ry = ti.randn(); rz = ti.randn()
        rn = ti.sqrt(rx**2 + ry**2 + rz**2) + 1e-6
        rx /= rn; ry /= rn; rz /= rn
        
        rvx = nx + rx
        rvy = ny + ry
        rvz = nz + rz
        
        r_norm = ti.sqrt(rvx**2 + rvy**2 + rvz**2) + 1e-6
        rvx /= r_norm; rvy /= r_norm; rvz /= r_norm
        
    return rvx, rvy, rvz

# ================= 3. 几何初始化 (3D深孔) =================

@ti.kernel
def init_grid():
    cx = SIZE_X / 2.0
    cy = SIZE_Y / 2.0
    angle_rad = 3 * math.pi / 180
    k_mask = ti.abs(ti.tan(angle_rad))
    
    for i, j, k in grid_exist:
        grid_count_cl[i, j, k] = 0
        
        if k <= vacuum:
            grid_exist[i, j, k] = 0.0
            grid_material[i, j, k] = 0
            
        elif vacuum < k < deep_border:
            offset = (deep_border - k) * k_mask
            current_radius = HOLE_RADIUS + offset
            dist = ti.sqrt((i - cx)**2 + (j - cy)**2)
            if dist < current_radius:
                grid_exist[i, j, k] = 0.0 
                grid_material[i, j, k] = 0
            else:
                grid_exist[i, j, k] = 1.0 
                grid_material[i, j, k] = 2
                
        else:
            grid_exist[i, j, k] = 1.0
            grid_material[i, j, k] = 1

# ================= 4. 核心仿真逻辑 (完全继承你的物理规则) =================

@ti.kernel
def simulate_batch():
    for _ in range(BATCH_SIZE):
        px = ti.random() * (SIZE_X - 1)
        py = ti.random() * (SIZE_Y - 1)
        pz = 1.0 
        
        is_ion = 0
        if ti.random() > RATIO: is_ion = 1
        
        sigma = (1.91 if is_ion==1 else 15.0) * (math.pi/180)
        theta = ti.abs(ti.randn() * sigma) 
        theta = ti.min(theta, math.pi/2 - 0.1)
        phi = ti.random() * 2.0 * math.pi 
        
        vx = ti.sin(theta) * ti.cos(phi)
        vy = ti.sin(theta) * ti.sin(phi)
        vz = ti.cos(theta) 
        
        alive = True
        steps = 0
        ref_count = 0  
        energy = 1.0  # 能量追踪
        
        while alive and steps < 3000:
            steps += 1
            step_size = 0.5
            px_n = px + vx * step_size
            py_n = py + vy * step_size
            pz_n = pz + vz * step_size
            
            if px_n < 0: px_n += SIZE_X
            if px_n >= SIZE_X: px_n -= SIZE_X
            if py_n < 0: py_n += SIZE_Y
            if py_n >= SIZE_Y: py_n -= SIZE_Y
            if pz_n < 0 or pz_n >= SIZE_Z: alive = False; break
            
            ipx, ipy, ipz = int(px_n), int(py_n), int(pz_n)
            
            if grid_exist[ipx, ipy, ipz] > 0.5:
                mat = grid_material[ipx, ipy, ipz]
                cl_n = grid_count_cl[ipx, ipy, ipz]
                
                nx, ny, nz = get_surface_normal_3d(ipx, ipy, ipz)
                cos_theta = -(vx * nx + vy * ny + vz * nz)
                cos_theta = ti.max(0.0, ti.min(1.0, cos_theta))
                theta_coll = ti.acos(cos_theta)

                did_reflect = False
                threshold = math.pi / 3  
                prob_reflect = 0.0 

                if is_ion == 1:
                    angle_else = ti.max(0.0, (theta_coll - threshold) / (math.pi/2 - threshold))
                    prob_reflect = ti.min(1.0, angle_else)
                    
                    if mat == 2: 
                        prob_reflect = ti.max(prob_reflect + 0.4, 0.85)
                    
                    if ti.random() < prob_reflect:
                        did_reflect = True
                        ref_count += 1
                else:
                    prob_reflect = 0.5 + 0.1 * cl_n 
                    if prob_reflect > 0.95: prob_reflect = 0.95
                    if ti.random() < 0.8:
                        did_reflect = True
                        ref_count += 1
                
                if did_reflect:
                    vx, vy, vz = get_reflection_vector_3d(vx, vy, vz, nx, ny, nz, is_ion)
                    px = px + nx * 1.5
                    py = py + ny * 1.5
                    pz = pz + nz * 1.5
                    
                    if is_ion == 1:
                        if mat == 2:
                            energy *= 0.95 
                        else:
                            energy *= 0.40 
                else:
                    if is_ion == 1:
                        prob_etch = 0.1 
                        if cl_n > 0: prob_etch += 0.2
                        
                        prob_etch *= energy 
                        if mat == 2: prob_etch *= 0.2 
                        
                        if ti.random() < prob_etch:
                            grid_exist[ipx, ipy, ipz] -= 0.3 
                            if grid_exist[ipx, ipy, ipz] <= 0.0:
                                grid_material[ipx, ipy, ipz] = 0
                            alive = False 
                        else:
                            alive = False 
                    else:
                        prob_etch = 0.0
                        if mat == 1 and cl_n >= 3:
                            prob_etch = 0.1 * 0.2
                        
                        if ti.random() < prob_etch:
                            grid_exist[ipx, ipy, ipz] -= 0.3
                            if grid_exist[ipx, ipy, ipz] <= 0.0:
                                grid_material[ipx, ipy, ipz] = 0
                                grid_count_cl[ipx, ipy, ipz] = 0
                            alive = False
                        else:
                            prob_adsorb = 1.0 - cl_n/4.0
                            if cl_n < 3 and ti.random() < prob_adsorb * 0.7:
                                grid_count_cl[ipx, ipy, ipz] += 1
                            alive = False
            else:
                px, py, pz = px_n, py_n, pz_n

# ================= 5. 3D 渲染数据准备 =================

@ti.kernel
def update_render_particles():
    """ 提取用于 3D 渲染的体素数据 """
    particle_cnt[None] = 0
    for i, j, k in grid_exist:
        if grid_exist[i, j, k] > 0.5:
            # 【重点：切西瓜视角】切掉前一半，让我们能看到孔洞内部！
            if j > SIZE_Y // 2: 
                continue
            
            # 表面检测：只渲染暴露在外的原子，极大节约显卡性能
            is_surface = False
            if k == 0 or k == SIZE_Z-1 or i == 0 or i == SIZE_X-1 or j == 0 or j == SIZE_Y // 2:
                is_surface = True
            else:
                if grid_exist[i+1,j,k]<0.5 or grid_exist[i-1,j,k]<0.5 or \
                   grid_exist[i,j+1,k]<0.5 or grid_exist[i,j-1,k]<0.5 or \
                   grid_exist[i,j,k+1]<0.5 or grid_exist[i,j,k-1]<0.5:
                    is_surface = True
                    
            if is_surface:
                idx = ti.atomic_add(particle_cnt[None], 1)
                
                # 映射到 3D 渲染坐标: X 和 Z 归一化，Y轴代表深度(反转)
                rx = (i / SIZE_X) - 0.5
                rz = (j / SIZE_Y) - 0.5
                ry = 1.0 - (k / SIZE_Z)  # GGUI中Y向上
                render_pos[idx] = ti.Vector([rx, ry, rz])
                
                # 颜色映射
                mat = grid_material[i, j, k]
                if mat == 1: # 硅基底 (深蓝色)
                    render_color[idx] = ti.Vector([0.1, 0.3, 0.8])
                elif mat == 2: # 掩膜 (青蓝色)
                    render_color[idx] = ti.Vector([0.0, 0.8, 0.8])

# ================= 6. 主程序 (启动 3D 引擎) =================

def main():
    print(">>> 正在初始化 3D 晶格...")
    init_grid()
    
    # 启动 Taichi 3D GUI 窗口
    window = ti.ui.Window("3D Deep Hole Etching", (1000, 1000), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    # 设置初始摄像机位置 (斜向下看中心点)
    camera.position(0.8, 1.2, 1.0)
    camera.lookat(0.0, 0.5, 0.0)
    
    print("\n===============================================")
    print(">>> 3D 渲染窗口已启动！")
    print(">>> 【操作指南】:")
    print(">>> 按住鼠标 [左键] 拖动：旋转视角")
    print(">>> 按住鼠标 [右键] 拖动：平移视角")
    print(">>> 滚动鼠标 [滚轮]：缩放")
    print("===============================================\n")
    
    step_count = 0
    while window.running:
        # 每帧跑 5 个 Batch 加速仿真
        for _ in range(5):
            simulate_batch()
            step_count += BATCH_SIZE
            
        # 更新用于渲染的点云
        update_render_particles()
        
        # 用户交互 (允许鼠标控制相机)
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        
        # 设置环境光和光源，营造真实立体感
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        
        # 渲染粒子体素
        scene.particles(render_pos, radius=0.003, per_vertex_color=render_color, index_count=particle_cnt[None])
        
        # 显示当前帧
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()