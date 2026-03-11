import taichi as ti
import numpy as np
from scipy.ndimage import gaussian_filter
import math
import os
import csv

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ================= 1. 环境初始化 =================
ti.init(arch=ti.gpu)

SAVE_DIR = r"Csv/Test_3D_DeepHole_FastView_2026"
os.makedirs(SAVE_DIR, exist_ok=True)

PI = math.pi

# ================= 2. 网格与仿真参数 =================
# 说明：
# NX, NY, NZ 分别对应 x / 深度y / z
# NY 是刻蚀深度方向（向下）

NX, NY, NZ = 192, 256, 192

VACUUM_Y = 24          # 顶部真空层厚度
MASK_BOTTOM = 56       # 掩膜层底部位置
HOLE_RADIUS = 26       # 初始圆孔半径
MASK_MARGIN = 18       # 边界保护带，避免侧边掩膜太薄

TOTAL_PARTICLES = 2_000_000
BATCH_SIZE = 2000
RATIO = 10.0 / 11.0    # 中性/总粒子比例

SMOOTH_INTERVAL = 2
MAX_STEPS = 2500
STEP_LEN = 1.0

# ================= 3. 显示参数（流畅版） =================
SIM_BATCHES_PER_FRAME = 4
DISPLAY_UPDATE_INTERVAL = 80
DISPLAY_DOWNSAMPLE = 2
DISPLAY_MAX_POINTS = 120000
POINT_RADIUS = 0.0032

# ================= 4. Taichi 数据场 =================
# 材料编号：0=真空, 1=Si, 2=Mask
grid_exist = ti.field(dtype=ti.f32, shape=(NX, NY, NZ))
grid_material = ti.field(dtype=ti.i32, shape=(NX, NY, NZ))
grid_count_cl = ti.field(dtype=ti.i32, shape=(NX, NY, NZ))
grid_temp = ti.field(dtype=ti.f32, shape=(NX, NY, NZ))

# 实时显示点云缓冲区
display_points = ti.Vector.field(3, dtype=ti.f32, shape=DISPLAY_MAX_POINTS)
display_colors = ti.Vector.field(3, dtype=ti.f32, shape=DISPLAY_MAX_POINTS)


# ================= 5. 物理辅助函数 =================

@ti.func
def wrap_x(a: float):
    b = a
    if b < 0:
        b += NX
    if b >= NX:
        b -= NX
    return b


@ti.func
def wrap_z(a: float):
    b = a
    if b < 0:
        b += NZ
    if b >= NZ:
        b -= NZ
    return b


@ti.func
def normalize3(x: float, y: float, z: float):
    n = ti.sqrt(x * x + y * y + z * z) + 1e-8
    return x / n, y / n, z / n


@ti.func
def get_surface_normal(px: int, py: int, pz: int):
    """3D 表面法线（指向真空）"""
    nx, ny, nz = 0.0, 0.0, 0.0

    for i, j, k in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
        if i != 0 or j != 0 or k != 0:
            xi = px + i
            yi = py + j
            zi = pz + k
            if 0 <= xi < NX and 0 <= yi < NY and 0 <= zi < NZ:
                if grid_exist[xi, yi, zi] < 0.5:
                    nx += float(i)
                    ny += float(j)
                    nz += float(k)

    norm = ti.sqrt(nx * nx + ny * ny + nz * nz)

    # Taichi 中不能在运行时 if 里提前 return，所以统一在最后 return
    outx, outy, outz = 0.0, -1.0, 0.0  # 默认朝上（朝真空）
    if norm >= 1e-6:
        inv = 1.0 / norm
        outx = nx * inv
        outy = ny * inv
        outz = nz * inv

    return outx, outy, outz


@ti.func
def get_reflection_vector(vx: float, vy: float, vz: float,
                          nx: float, ny: float, nz: float,
                          is_ion: int):
    """3D 反射向量：离子镜面反射；中性粒子 Lambertian 半球漫反射"""
    rvx, rvy, rvz = 0.0, 0.0, 0.0

    if is_ion == 1:
        # 镜面反射
        dot = vx * nx + vy * ny + vz * nz
        rvx = vx - 2.0 * dot * nx
        rvy = vy - 2.0 * dot * ny
        rvz = vz - 2.0 * dot * nz
    else:
        # 构造与法线正交的一组基
        ax, ay, az = 0.0, 1.0, 0.0
        if ti.abs(ny) > 0.9:
            ax, ay, az = 1.0, 0.0, 0.0

        # t1 = a x n
        t1x = ay * nz - az * ny
        t1y = az * nx - ax * nz
        t1z = ax * ny - ay * nx
        t1x, t1y, t1z = normalize3(t1x, t1y, t1z)

        # t2 = n x t1
        t2x = ny * t1z - nz * t1y
        t2y = nz * t1x - nx * t1z
        t2z = nx * t1y - ny * t1x
        t2x, t2y, t2z = normalize3(t2x, t2y, t2z)

        u1 = ti.random()
        u2 = ti.random()
        phi = 2.0 * PI * u2

        # Lambertian 半球采样
        cos_theta = ti.sqrt(u1)
        sin_theta = ti.sqrt(1.0 - u1)

        rvx = nx * cos_theta + t1x * ti.cos(phi) * sin_theta + t2x * ti.sin(phi) * sin_theta
        rvy = ny * cos_theta + t1y * ti.cos(phi) * sin_theta + t2y * ti.sin(phi) * sin_theta
        rvz = nz * cos_theta + t1z * ti.cos(phi) * sin_theta + t2z * ti.sin(phi) * sin_theta

    return normalize3(rvx, rvy, rvz)


@ti.kernel
def smooth_grid():
    """3D 表面平滑，减弱体素噪声"""
    w_center = 0.94
    w_nb = (1.0 - w_center) / 6.0

    for i, j, k in grid_exist:
        if 1 <= i < NX - 1 and 1 <= j < NY - 1 and 1 <= k < NZ - 1:
            val = (
                grid_exist[i, j, k] * w_center
                + (grid_exist[i + 1, j, k] + grid_exist[i - 1, j, k]
                   + grid_exist[i, j + 1, k] + grid_exist[i, j - 1, k]
                   + grid_exist[i, j, k + 1] + grid_exist[i, j, k - 1]) * w_nb
            )
            grid_temp[i, j, k] = val
        else:
            grid_temp[i, j, k] = grid_exist[i, j, k]

    for i, j, k in grid_exist:
        grid_exist[i, j, k] = grid_temp[i, j, k]


# ================= 6. 3D 深孔几何初始化 =================

@ti.kernel
def init_grid():
    cx = NX * 0.5
    cz = NZ * 0.5

    for i, j, k in grid_exist:
        grid_count_cl[i, j, k] = 0

        if j < VACUUM_Y:
            # 顶部真空
            grid_exist[i, j, k] = 0.0
            grid_material[i, j, k] = 0

        elif j < MASK_BOTTOM:
            # 掩膜层：中间开一个圆孔
            dx = i - cx
            dz = k - cz
            inside_hole = (dx * dx + dz * dz) <= HOLE_RADIUS * HOLE_RADIUS

            if inside_hole:
                grid_exist[i, j, k] = 0.0
                grid_material[i, j, k] = 0
            else:
                if (MASK_MARGIN <= i < NX - MASK_MARGIN and
                        MASK_MARGIN <= k < NZ - MASK_MARGIN):
                    grid_exist[i, j, k] = 1.0
                    grid_material[i, j, k] = 2
                else:
                    grid_exist[i, j, k] = 0.0
                    grid_material[i, j, k] = 0

        else:
            # 底下是 Si
            grid_exist[i, j, k] = 1.0
            grid_material[i, j, k] = 1


# ================= 7. 核心仿真逻辑（反应机理与反射机制保持不变） =================

@ti.kernel
def simulate_batch():
    for _ in range(BATCH_SIZE):
        # --- A. 粒子从顶面随机入射 ---
        px = ti.random() * (NX - 1)
        py = 1.0
        pz = ti.random() * (NZ - 1)

        is_ion = 0
        if ti.random() > RATIO:
            is_ion = 1

        sigma = (1.91 if is_ion == 1 else 7.64) * (PI / 180.0)

        # 3D 入射方向：围绕竖直方向形成高斯角分布
        sx = ti.randn() * sigma
        sz = ti.randn() * sigma
        vx, vy, vz = normalize3(sx, 1.0, sz)

        alive = True
        steps = 0
        ref_count = 0

        while alive and steps < MAX_STEPS:
            steps += 1

            px_n = px + vx * STEP_LEN
            py_n = py + vy * STEP_LEN
            pz_n = pz + vz * STEP_LEN

            px_n = wrap_x(px_n)
            pz_n = wrap_z(pz_n)

            if py_n < 0 or py_n >= NY:
                alive = False
                break

            ipx, ipy, ipz = int(px_n), int(py_n), int(pz_n)

            # --- B. 碰撞检测 ---
            if grid_exist[ipx, ipy, ipz] > 0.5:
                mat = grid_material[ipx, ipy, ipz]
                cl_n = grid_count_cl[ipx, ipy, ipz]

                nx, ny, nz = get_surface_normal(ipx, ipy, ipz)

                cos_theta = -(vx * nx + vy * ny + vz * nz)
                cos_theta = ti.max(0.0, ti.min(1.0, cos_theta))
                theta_coll = ti.acos(cos_theta)

                did_reflect = False
                threshold = PI / 3.0
                prob_reflect = 0.0

                # --- 1. 反射判定 ---
                if is_ion == 1:
                    angle_else = ti.max(0.0, (theta_coll - threshold) / (PI / 2.0 - threshold))
                    angle_else = ti.min(1.0, angle_else)
                    prob_reflect = angle_else
                    if mat == 2:
                        prob_reflect += 0.2

                    if ti.random() < prob_reflect:
                        did_reflect = True
                        ref_count += 1

                else:
                    # 保持你原程序的实际判定逻辑
                    prob_reflect = 0.5 + 0.1 * cl_n
                    if prob_reflect > 0.95:
                        prob_reflect = 0.95

                    if ti.random() < 0.8:
                        did_reflect = True
                        ref_count += 1

                # --- 2. 行为分支 ---
                if did_reflect:
                    vx, vy, vz = get_reflection_vector(vx, vy, vz, nx, ny, nz, is_ion)
                    px = wrap_x(px + nx * 1.5)
                    py = py + ny * 1.5
                    pz = wrap_z(pz + nz * 1.5)

                    if py < 0 or py >= NY:
                        alive = False

                else:
                    # --- 反应判定 ---
                    if is_ion == 1:
                        # 离子刻蚀
                        prob_etch = 0.1
                        if cl_n > 0:
                            prob_etch += 0.2

                        # 反射后的离子能量衰减，刻蚀能力下降
                        if ref_count >= 1:
                            prob_etch *= 0.1

                        # 掩膜更难被刻蚀
                        if mat == 2:
                            prob_etch *= 0.2

                        if ti.random() < prob_etch:
                            grid_exist[ipx, ipy, ipz] = ti.max(0.0, grid_exist[ipx, ipy, ipz] - 0.2)
                            if grid_exist[ipx, ipy, ipz] <= 0.0:
                                grid_material[ipx, ipy, ipz] = 0
                            alive = False
                        else:
                            alive = False

                    else:
                        # 中性粒子反应
                        prob_etch = 0.0
                        if mat == 1 and cl_n >= 3:
                            prob_etch = 0.1 * 0.1

                        if ti.random() < prob_etch:
                            grid_exist[ipx, ipy, ipz] = ti.max(0.0, grid_exist[ipx, ipy, ipz] - 0.2)
                            if grid_exist[ipx, ipy, ipz] <= 0.0:
                                grid_material[ipx, ipy, ipz] = 0
                                grid_count_cl[ipx, ipy, ipz] = 0
                            alive = False
                        else:
                            # 吸附
                            prob_adsorb = 1.0 - cl_n / 4.0
                            if cl_n < 3 and ti.random() < prob_adsorb * 0.1:
                                grid_count_cl[ipx, ipy, ipz] += 1
                            alive = False
            else:
                px, py, pz = px_n, py_n, pz_n


# ================= 8. 轻量级实时显示：表面点云 =================

def build_surface_point_cloud(exist_data, mat_data,
                              ds=DISPLAY_DOWNSAMPLE,
                              max_points=DISPLAY_MAX_POINTS):
    """
    从体素中提取表面点云，供 Taichi GGUI 实时显示。
    比 marching_cubes + matplotlib 快三很多。
    """
    solid = exist_data[::ds, ::ds, ::ds] > 0.5
    mats = mat_data[::ds, ::ds, ::ds]

    if solid.size == 0:
        pts = np.full((DISPLAY_MAX_POINTS, 3), 1e5, dtype=np.float32)
        cols = np.zeros((DISPLAY_MAX_POINTS, 3), dtype=np.float32)
        return pts, cols

    # 表面判断：当前是 solid，且六邻域至少有一个是空
    p = np.pad(solid, 1, mode='constant', constant_values=False)
    c = p[1:-1, 1:-1, 1:-1]

    surface = c & (
        (~p[:-2, 1:-1, 1:-1]) |
        (~p[2:, 1:-1, 1:-1]) |
        (~p[1:-1, :-2, 1:-1]) |
        (~p[1:-1, 2:, 1:-1]) |
        (~p[1:-1, 1:-1, :-2]) |
        (~p[1:-1, 1:-1, 2:])
    )

    idx = np.argwhere(surface)

    if len(idx) == 0:
        pts = np.full((DISPLAY_MAX_POINTS, 3), 1e5, dtype=np.float32)
        cols = np.zeros((DISPLAY_MAX_POINTS, 3), dtype=np.float32)
        return pts, cols

    # 点太多就随机抽样
    if len(idx) > max_points:
        pick = np.random.choice(len(idx), size=max_points, replace=False)
        idx = idx[pick]

    sampled_mat = mats[idx[:, 0], idx[:, 1], idx[:, 2]]

    max_dim = float(max(NX, NY, NZ))
    pts_valid = np.empty((len(idx), 3), dtype=np.float32)

    # 显示坐标：把 y(深度方向)映射为竖直向下
    pts_valid[:, 0] = (idx[:, 0] * ds - NX * 0.5) / max_dim
    pts_valid[:, 1] = (NY * 0.22 - idx[:, 1] * ds) / max_dim
    pts_valid[:, 2] = (idx[:, 2] * ds - NZ * 0.5) / max_dim

    cols_valid = np.empty((len(idx), 3), dtype=np.float32)
    si_mask = sampled_mat == 1
    mask_mask = sampled_mat == 2

    cols_valid[si_mask] = np.array([0.05, 0.12, 0.70], dtype=np.float32)   # Si
    cols_valid[mask_mask] = np.array([0.00, 0.95, 0.95], dtype=np.float32) # Mask

    pts = np.full((DISPLAY_MAX_POINTS, 3), 1e5, dtype=np.float32)
    cols = np.zeros((DISPLAY_MAX_POINTS, 3), dtype=np.float32)

    n = len(pts_valid)
    pts[:n] = pts_valid
    cols[:n] = cols_valid

    return pts, cols


def update_display_buffers(exist_data, mat_data):
    pts, cols = build_surface_point_cloud(exist_data, mat_data)
    display_points.from_numpy(pts)
    display_colors.from_numpy(cols)


# ================= 9. 最终导出高质量表面 =================

def extract_mesh(binary_volume, sigma=1.0, step_size=1):
    """
    仅用于最终导出高质量表面，不用于实时显示。
    """
    if not HAS_SKIMAGE:
        return None, None

    vol = gaussian_filter(binary_volume.astype(np.float32), sigma=sigma)
    vmin, vmax = vol.min(), vol.max()
    if not (vmin < 0.5 < vmax):
        return None, None

    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, step_size=step_size)
    return verts, faces


def save_surface_csv(verts, filename):
    if verts is None or len(verts) == 0:
        return

    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y_depth", "z"])
        writer.writerows(verts.tolist())


# ================= 10. 主程序 =================

def main():
    init_grid()

    window = ti.ui.Window("3D Deep Hole Etching - Fast View", (1280, 820), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    camera.position(1.8, 1.2, 1.8)
    camera.lookat(0.0, -0.15, 0.0)
    camera.up(0.0, 1.0, 0.0)

    print(">>> 3D 深孔刻蚀模拟开始（流畅版实时三维显示） <<<")
    print("操作方式：按住鼠标右键拖动旋转，W/A/S/D/E/Q 移动视角。")

    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    batch_id = 0

    # 初始化第一帧
    ti.sync()
    exist_data = grid_exist.to_numpy()
    mat_data = grid_material.to_numpy()
    update_display_buffers(exist_data, mat_data)

    # ---------- 仿真阶段 ----------
    while window.running and batch_id < num_batches:
        for _ in range(SIM_BATCHES_PER_FRAME):
            if batch_id >= num_batches:
                break

            simulate_batch()

            if batch_id % SMOOTH_INTERVAL == 0:
                smooth_grid()

            batch_id += 1

        # 不是每一帧都更新显示
        if (batch_id % DISPLAY_UPDATE_INTERVAL == 0) or (batch_id >= num_batches):
            ti.sync()
            exist_data = grid_exist.to_numpy()
            mat_data = grid_material.to_numpy()
            update_display_buffers(exist_data, mat_data)
            print(f"进度: {batch_id / max(1, num_batches):.1%}", end="\r")

        # 相机交互
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

        # 场景绘制
        canvas.set_background_color((0.06, 0.08, 0.12))
        scene.set_camera(camera)
        scene.ambient_light((0.78, 0.78, 0.78))
        scene.point_light(pos=(2.0, 2.5, 2.0), color=(1.0, 1.0, 1.0))
        scene.particles(display_points, per_vertex_color=display_colors, radius=POINT_RADIUS)

        canvas.scene(scene)
        window.show()

    print("\n>>> 模拟完成，开始导出最终表面。")

    # 最终结果导出
    ti.sync()
    exist_data = grid_exist.to_numpy()
    mat_data = grid_material.to_numpy()
    update_display_buffers(exist_data, mat_data)

    if HAS_SKIMAGE:
        final_si = ((exist_data > 0.5) & (mat_data == 1)).astype(np.float32)
        verts, _ = extract_mesh(final_si, sigma=0.8, step_size=1)
        save_surface_csv(verts, "surface_final.csv")
        print(f"已导出: {os.path.join(SAVE_DIR, 'surface_final.csv')}")
    else:
        print("未安装 scikit-image，跳过最终 marching_cubes 导出。")

    print(">>> 现在进入最终结果查看模式，关闭窗口即可结束程序。")

    # ---------- 最终查看阶段 ----------
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

        canvas.set_background_color((0.06, 0.08, 0.12))
        scene.set_camera(camera)
        scene.ambient_light((0.78, 0.78, 0.78))
        scene.point_light(pos=(2.0, 2.5, 2.0), color=(1.0, 1.0, 1.0))
        scene.particles(display_points, per_vertex_color=display_colors, radius=POINT_RADIUS)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()