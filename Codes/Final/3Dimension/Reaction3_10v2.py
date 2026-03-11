import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
import math
import os
import csv

# ================= 1. 环境初始化 =================
ti.init(arch=ti.gpu)

SAVE_DIR = r"Csv/Test_MarchingCubes_3D_2026"
os.makedirs(SAVE_DIR, exist_ok=True)

PI = math.pi

# ----------------- 3D 网格尺寸 -----------------
# 注意：3D 不能再直接沿用 1000x700 的二维尺度，否则显存和运算量会非常大。
# 这组尺寸更适合先跑通 3D 深孔刻蚀。
NX, NY, NZ = 192, 256, 192   # x / 深度y / z

VACUUM_Y = 24                # 顶部真空层厚度
MASK_BOTTOM = 56             # 掩膜底部位置
HOLE_RADIUS = 26             # 初始圆孔半径（深孔开口）
MASK_MARGIN = 18             # 边界保护带，避免边缘掩膜太薄

TOTAL_PARTICLES = 2_000
BATCH_SIZE = 2000
RATIO = 10.0 / 11.0          # 中性/总粒子比例
VIS_INTERVAL = 40
SMOOTH_INTERVAL = 2
MAX_STEPS = 2500
STEP_LEN = 1.0

# 材料编号：0=真空, 1=Si, 2=Mask
grid_exist = ti.field(dtype=ti.f32, shape=(NX, NY, NZ))
grid_material = ti.field(dtype=ti.i32, shape=(NX, NY, NZ))
grid_count_cl = ti.field(dtype=ti.i32, shape=(NX, NY, NZ))
grid_temp = ti.field(dtype=ti.f32, shape=(NX, NY, NZ))


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


# ================= 2. 物理辅助函数 =================

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

    outx, outy, outz = 0.0, -1.0, 0.0  # 默认朝上（指向真空）
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


# ================= 3. 3D 深孔几何初始化 =================

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
                if (MASK_MARGIN <= i and i < NX - MASK_MARGIN and
                    MASK_MARGIN <= k and k < NZ - MASK_MARGIN):
                    grid_exist[i, j, k] = 1.0
                    grid_material[i, j, k] = 2
                else:
                    grid_exist[i, j, k] = 0.0
                    grid_material[i, j, k] = 0

        else:
            # 底下全部是 Si
            grid_exist[i, j, k] = 1.0
            grid_material[i, j, k] = 1


# ================= 4. 核心仿真逻辑（3D；机制保持不变） =================

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

        # 2D 原来只有一个偏角；3D 中扩展成两个横向偏角，角宽仍保持同一个 sigma
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

                # 法线指向真空；入射粒子指向固体，所以取 -(v·n)
                cos_theta = -(vx * nx + vy * ny + vz * nz)
                cos_theta = ti.max(0.0, ti.min(1.0, cos_theta))
                theta_coll = ti.acos(cos_theta)

                did_reflect = False

                # --- 1. 反射判定（与你原程序一致） ---
                threshold = PI / 3.0
                prob_reflect = 0.0

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
                    # 仍保持你原代码的实际判定方式：中性固定 0.8 反射
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
                    # --- 反应判定（与你原程序一致） ---
                    if is_ion == 1:
                        # 离子刻蚀
                        prob_etch = 0.1
                        if cl_n > 0:
                            prob_etch += 0.2

                        # 反射过的离子能量衰减 -> 刻蚀概率降低
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


# ================= 5. 3D 表面提取与可视化 =================

def extract_mesh(binary_volume, sigma=1.0, step_size=2):
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


def plot_mesh(ax, verts, faces, face_rgba):
    if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
        return

    # 数组轴顺序是 (x, y_depth, z)
    # 显示时把 depth 映射到 z 轴，便于看“向下刻蚀”
    show_verts = np.stack([verts[:, 0], verts[:, 2], verts[:, 1]], axis=1)
    tri = Poly3DCollection(show_verts[faces], alpha=face_rgba[3], linewidths=0.0)
    tri.set_facecolor(face_rgba)
    tri.set_edgecolor((0, 0, 0, 0))
    ax.add_collection3d(tri)


def fallback_voxel_plot(ax, exist_data, mat_data, ds=4):
    solid = exist_data[::ds, ::ds, ::ds] > 0.5
    mats = mat_data[::ds, ::ds, ::ds]

    colors = np.empty(solid.shape + (4,), dtype=np.float32)
    colors[..., :] = (0.0, 0.0, 0.0, 0.0)
    colors[(solid) & (mats == 2)] = (0.0, 1.0, 1.0, 0.25)   # mask
    colors[(solid) & (mats == 1)] = (0.0, 0.1, 0.55, 0.55)  # silicon

    ax.voxels(solid, facecolors=colors, edgecolor=None)


def render_3d(ax, exist_data, mat_data, title_text):
    ax.cla()

    if HAS_SKIMAGE:
        mask_vol = ((exist_data > 0.5) & (mat_data == 2)).astype(np.float32)
        si_vol = ((exist_data > 0.5) & (mat_data == 1)).astype(np.float32)

        mask_verts, mask_faces = extract_mesh(mask_vol, sigma=0.8, step_size=2)
        si_verts, si_faces = extract_mesh(si_vol, sigma=0.8, step_size=2)

        plot_mesh(ax, mask_verts, mask_faces, (0.0, 1.0, 1.0, 0.18))
        plot_mesh(ax, si_verts, si_faces, (0.0, 0.1, 0.55, 0.60))

        if si_verts is not None:
            save_surface_csv(si_verts, "surface_latest.csv")
    else:
        fallback_voxel_plot(ax, exist_data, mat_data, ds=4)

    ax.set_xlim(0, NX)
    ax.set_ylim(0, NZ)
    ax.set_zlim(NY, 0)  # 深度朝下
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Depth (Y)")
    ax.view_init(elev=24, azim=-58)
    ax.set_title(title_text)


# ================= 6. 主程序 =================

def main():
    if not HAS_SKIMAGE:
        print("Warning: 未检测到 scikit-image，将使用 downsample voxel 显示。")
        print("建议安装: pip install scikit-image")

    init_grid()

    plt.ion()
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    print(">>> 3D 深孔刻蚀模拟开始（反射优先 + Marching Cubes 三维显示） <<<")

    num_batches = TOTAL_PARTICLES // BATCH_SIZE

    for i in range(num_batches):
        simulate_batch()

        if i % SMOOTH_INTERVAL == 0:
            smooth_grid()

        if i % VIS_INTERVAL == 0:
            ti.sync()

            exist_data = grid_exist.to_numpy()
            mat_data = grid_material.to_numpy()

            render_3d(
                ax,
                exist_data,
                mat_data,
                f"3D Deep Hole Etching: {i * BATCH_SIZE}/{TOTAL_PARTICLES}"
            )

            plt.pause(0.01)
            print(f"进度: {i / max(1, num_batches):.1%}", end="\r")

    ti.sync()
    exist_data = grid_exist.to_numpy()
    mat_data = grid_material.to_numpy()
    render_3d(ax, exist_data, mat_data, "3D Deep Hole Etching: Final")

    if HAS_SKIMAGE:
        final_si = ((exist_data > 0.5) & (mat_data == 1)).astype(np.float32)
        verts, _ = extract_mesh(final_si, sigma=0.8, step_size=1)
        save_surface_csv(verts, "surface_final.csv")

    print("\n>>> 模拟完成。")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()