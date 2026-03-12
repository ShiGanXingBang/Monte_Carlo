"""
尝试将bowing现象明显化,使Mask和Substrate中间的过渡更加平滑。
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter
import math
import os
import csv

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ============================================================
# 1. 环境初始化
# ============================================================
ti.init(arch=ti.gpu)

SAVE_DIR = r"Csv/Bowing_SEM_Reproduction_2026_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ 计算域与几何参数 ------------------
ROWS, COLS = 1000, 700
VACUUM_Y = 100
MASK_BOTTOM_Y = 230

LEFT_BORDER = 170
RIGHT_BORDER = 240
SPACE = 80
NUM_TRENCH = 4
CD = RIGHT_BORDER - LEFT_BORDER
MASK_TAPER_DEG = 4.0

# ------------------ 粒子统计参数 ------------------
TOTAL_PARTICLES = 20_000_000
BATCH_SIZE = 4000
RATIO_NEUTRAL = 10.0 / 11.0
MAX_STEPS = 3200
STEP_LEN = 1.10
DISPLAY_EVERY = 50
MAX_HISTORY = 90

# ------------------ 角分布与主刻蚀参数 ------------------
ION_SIGMA_DEG = 2.2
NEUTRAL_SIGMA_DEG = 7.5

ION_BASE_ETCH = 0.10
ION_CL_BOOST = 0.16
ION_REFLECT_FACTOR_1 = 0.60
ION_REFLECT_FACTOR_2 = 0.35

# ------------------ Mask 有限刻蚀 + 角点增强 ------------------
MASK_ETCH_BASE = 0.008
MASK_ETCH_ANGLE_GAIN = 0.020
MASK_CORNER_MULTIPLIER = 2.50
MASK_REFLECT_FACTOR_1 = 0.75
MASK_REFLECT_FACTOR_2 = 0.55
ETCH_STEP_MASK = 0.10

# ------------------ 中性粒子：吸附/反射/化学刻蚀 ------------------
NEUTRAL_REFLECT_BASE = 0.35
NEUTRAL_REFLECT_CL_GAIN = 0.08
NEUTRAL_REFLECT_MASK_GAIN = 0.20
NEUTRAL_REFLECT_CAP = 0.82
ADSORB_BASE = 0.22
MASK_ADSORB_FACTOR = 0.08
MAX_CL_COVERAGE = 4

# ------------------ 深度衰减：抑制“整段侧壁等强 bowing” ------------------
DEPTH_DECAY_LENGTH = 220.0
DEPTH_DECAY_STRENGTH = 0.45  # 底部化学侧蚀削弱到 ~55%

# ------------------ 数值平滑 ------------------
ENABLE_WEAK_SMOOTH = False
SMOOTH_EVERY = 20 * DISPLAY_EVERY
SMOOTH_CENTER = 0.995

# ------------------ 单次刻蚀步长 ------------------
ETCH_STEP_ION = 0.20
ETCH_STEP_NEUTRAL = 0.20

# ------------------ 视野窗口 ------------------
VIEW_MARGIN_X = 70
VIEW_MARGIN_Y_TOP = 20
VIEW_MARGIN_Y_BOTTOM = 320

# ============================================================
# 2. Taichi 数据场
# ============================================================
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))   # 0=真空, 1=Si, 2=Mask
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))
grid_temp = ti.field(dtype=ti.f32, shape=(ROWS, COLS))

# ============================================================
# 3. 物理辅助函数
# ============================================================
@ti.func
def clamp01(x: ti.f32) -> ti.f32:
    return ti.min(1.0, ti.max(0.0, x))


@ti.func
def get_surface_normal(px: int, py: int):
    """计算局部表面法线，指向真空。"""
    nx, ny = 0.0, 0.0
    for i, j in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        if 0 <= px + i < ROWS and 0 <= py + j < COLS:
            if grid_exist[px + i, py + j] < 0.5:
                nx += float(i)
                ny += float(j)
    norm = ti.sqrt(nx * nx + ny * ny) + 1e-6
    return nx / norm, ny / norm


@ti.func
def get_reflection_vector(vx: ti.f32, vy: ti.f32, nx: ti.f32, ny: ti.f32, is_ion: ti.i32):
    """离子镜面反射；中性粒子近似 Lambertian 漫反射。"""
    rvx, rvy = 0.0, 0.0
    if is_ion == 1:
        dot = vx * nx + vy * ny
        rvx = vx - 2.0 * dot * nx
        rvy = vy - 2.0 * dot * ny
    else:
        tx, ty = -ny, nx
        rand_s = (ti.random() - 0.5) * 2.0
        rand_s = ti.max(-1.0, ti.min(1.0, rand_s))
        rand_c = ti.sqrt(ti.max(0.0, 1.0 - rand_s * rand_s))
        rvx = nx * rand_c + tx * rand_s
        rvy = ny * rand_c + ty * rand_s

    norm = ti.sqrt(rvx * rvx + rvy * rvy) + 1e-6
    return rvx / norm, rvy / norm


@ti.func
def neutral_etch_prob(cl_n: ti.i32) -> ti.f32:
    """Cl 覆盖度越高，中性化学刻蚀越强。"""
    p = 0.0
    if cl_n == 0:
        p = 0.00
    elif cl_n == 1:
        p = 0.02
    elif cl_n == 2:
        p = 0.06
    elif cl_n == 3:
        p = 0.11
    else:
        p = 0.16
    return p


@ti.func
def is_mask_corner_or_foot(px: int, py: int) -> ti.i32:
    result = 0
    has_vac = 0
    has_si = 0

    if grid_material[px, py] == 2:
        for i, j in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            nx = px + i
            ny = py + j
            if 0 <= nx < ROWS and 0 <= ny < COLS:
                mat = grid_material[nx, ny]
                if mat == 0:
                    has_vac = 1
                elif mat == 1:
                    has_si = 1

        if has_vac == 1 and has_si == 1:
            result = 1

    return result


@ti.func
def mask_etch_prob(cos_theta: ti.f32, is_corner: ti.i32) -> ti.f32:
    """Mask 有限溅射：角部/脚部增强。"""
    p = MASK_ETCH_BASE + MASK_ETCH_ANGLE_GAIN * (1.0 - cos_theta)
    if is_corner == 1:
        p *= MASK_CORNER_MULTIPLIER
    return clamp01(p)


@ti.func
def depth_factor(py: int) -> ti.f32:
    """越往下，中性侧向化学刻蚀越弱，避免整段等强 bowing。"""
    d = (float(py) - float(MASK_BOTTOM_Y)) / DEPTH_DECAY_LENGTH
    d = clamp01(d)
    return 1.0 - DEPTH_DECAY_STRENGTH * d


@ti.kernel
def smooth_grid():
    """弱平滑：只在噪声太大时低频启用。"""
    w_center = SMOOTH_CENTER
    w_neighbor = (1.0 - w_center) / 4.0
    for i, j in grid_exist:
        if 1 <= i < ROWS - 1 and 1 <= j < COLS - 1:
            grid_temp[i, j] = (
                grid_exist[i, j] * w_center
                + (grid_exist[i + 1, j] + grid_exist[i - 1, j] + grid_exist[i, j + 1] + grid_exist[i, j - 1]) * w_neighbor
            )
        else:
            grid_temp[i, j] = grid_exist[i, j]

    for i, j in grid_exist:
        grid_exist[i, j] = grid_temp[i, j]


# ============================================================
# 4. 几何初始化
# ============================================================
@ti.kernel
def init_grid():
    k_mask = ti.abs(ti.tan(MASK_TAPER_DEG * math.pi / 180.0))

    for i, j in grid_exist:
        grid_count_cl[i, j] = 0

        if j <= VACUUM_Y:
            grid_exist[i, j] = 0.0
            grid_material[i, j] = 0
        elif j < MASK_BOTTOM_Y:
            grid_exist[i, j] = 1.0
            grid_material[i, j] = 2

            for n in range(NUM_TRENCH):
                x_left = LEFT_BORDER + n * (CD + SPACE)
                x_right = RIGHT_BORDER + n * (CD + SPACE)
                offset = int((MASK_BOTTOM_Y - j) * k_mask)
                open_left = max(0, x_left - offset)
                open_right = min(ROWS - 1, x_right + offset)

                # 掩膜脚部轻微预收缩，避免绝对理想直角
                local_shrink = 0
                if MASK_BOTTOM_Y - 8 <= j < MASK_BOTTOM_Y:
                    local_shrink = int((j - (MASK_BOTTOM_Y - 8)) * 0.5)

                if (open_left + local_shrink) < i < (open_right - local_shrink):
                    grid_exist[i, j] = 0.0
                    grid_material[i, j] = 0
        else:
            grid_exist[i, j] = 1.0
            grid_material[i, j] = 1


# ============================================================
# 5. 核心仿真逻辑（Mask 脚部刻蚀 + 角点增强 + 深度衰减）
# ============================================================
@ti.kernel
def simulate_batch():
    for k in range(BATCH_SIZE):
        px, py = ti.random() * (ROWS - 1), 1.0

        is_ion = 0
        if ti.random() > RATIO_NEUTRAL:
            is_ion = 1

        sigma = (ION_SIGMA_DEG if is_ion == 1 else NEUTRAL_SIGMA_DEG) * (math.pi / 180.0)
        angle = ti.randn() * sigma
        angle = ti.max(-math.pi / 2.0, ti.min(math.pi / 2.0, angle))

        vx, vy = ti.sin(angle), ti.cos(angle)
        alive = True
        steps = 0
        ref_count = 0

        while alive and steps < MAX_STEPS:
            steps += 1

            px_n = px + vx * STEP_LEN
            py_n = py + vy * STEP_LEN

            if px_n < 0:
                px_n += ROWS
            if px_n >= ROWS:
                px_n -= ROWS
            if py_n < 0 or py_n >= COLS:
                alive = False
                break

            ipx, ipy = int(px_n), int(py_n)

            if grid_exist[ipx, ipy] > 0.5:
                mat = grid_material[ipx, ipy]
                cl_n = grid_count_cl[ipx, ipy]
                if cl_n > MAX_CL_COVERAGE:
                    cl_n = MAX_CL_COVERAGE

                nx, ny = get_surface_normal(ipx, ipy)
                cos_theta = clamp01(-(vx * nx + vy * ny))

                prob_reflect = 0.0
                did_reflect = False
                if is_ion == 1:
                    prob_reflect = 0.12 + 0.70 * (1.0 - cos_theta)
                    if mat == 2:
                        prob_reflect += 0.10
                    prob_reflect = clamp01(prob_reflect)
                else:
                    prob_reflect = NEUTRAL_REFLECT_BASE + NEUTRAL_REFLECT_CL_GAIN * cl_n
                    if mat == 2:
                        prob_reflect += NEUTRAL_REFLECT_MASK_GAIN
                    if prob_reflect > NEUTRAL_REFLECT_CAP:
                        prob_reflect = NEUTRAL_REFLECT_CAP

                if ti.random() < prob_reflect:
                    did_reflect = True
                    ref_count += 1

                if did_reflect:
                    vx, vy = get_reflection_vector(vx, vy, nx, ny, is_ion)
                    px = px_n + nx * 1.2
                    py = py_n + ny * 1.2
                else:
                    if is_ion == 1:
                        if mat == 1:
                            angle_factor = 0.55 + 0.45 * cos_theta
                            prob_etch = ION_BASE_ETCH * angle_factor
                            prob_etch += ION_CL_BOOST * (cl_n / float(MAX_CL_COVERAGE))

                            if ref_count == 1:
                                prob_etch *= ION_REFLECT_FACTOR_1
                            elif ref_count >= 2:
                                prob_etch *= ION_REFLECT_FACTOR_2

                            prob_etch = clamp01(prob_etch)
                            if ti.random() < prob_etch:
                                grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - ETCH_STEP_ION)
                                if grid_exist[ipx, ipy] <= 0.0:
                                    grid_material[ipx, ipy] = 0
                                    grid_count_cl[ipx, ipy] = 0

                        elif mat == 2:
                            corner_flag = is_mask_corner_or_foot(ipx, ipy)
                            prob_mask = mask_etch_prob(cos_theta, corner_flag)

                            if ref_count == 1:
                                prob_mask *= MASK_REFLECT_FACTOR_1
                            elif ref_count >= 2:
                                prob_mask *= MASK_REFLECT_FACTOR_2

                            prob_mask = clamp01(prob_mask)
                            if ti.random() < prob_mask:
                                grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - ETCH_STEP_MASK)
                                if grid_exist[ipx, ipy] <= 0.0:
                                    grid_material[ipx, ipy] = 0
                                    grid_count_cl[ipx, ipy] = 0

                        alive = False

                    else:
                        prob_etch = 0.0
                        if mat == 1:
                            prob_etch = neutral_etch_prob(cl_n) * depth_factor(ipy)

                        if ti.random() < prob_etch:
                            grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - ETCH_STEP_NEUTRAL)
                            if grid_exist[ipx, ipy] <= 0.0:
                                grid_material[ipx, ipy] = 0
                                grid_count_cl[ipx, ipy] = 0
                        else:
                            prob_adsorb = ADSORB_BASE * (1.0 - cl_n / float(MAX_CL_COVERAGE))
                            prob_adsorb *= depth_factor(ipy)
                            if mat == 2:
                                prob_adsorb *= MASK_ADSORB_FACTOR
                            prob_adsorb = ti.max(0.0, prob_adsorb)

                            if grid_material[ipx, ipy] == 1 and grid_count_cl[ipx, ipy] < MAX_CL_COVERAGE:
                                if ti.random() < prob_adsorb:
                                    grid_count_cl[ipx, ipy] += 1
                        alive = False
            else:
                px, py = px_n, py_n


# ============================================================
# 6. 轮廓提取与保存
# ============================================================
def get_contours(raw_grid):
    smoothed = gaussian_filter(raw_grid.astype(np.float32), sigma=1.0)
    contour_list = []

    if HAS_SKIMAGE:
        contours = measure.find_contours(smoothed, 0.5)
        for contour in contours:
            one_contour = [(float(p[0]), float(p[1])) for p in contour]
            if len(one_contour) >= 12:
                xs = [p[0] for p in one_contour]
                ys = [p[1] for p in one_contour]
                if (max(xs) - min(xs) >= 2.0) or (max(ys) - min(ys) >= 2.0):
                    contour_list.append(one_contour)
    else:
        fig_tmp = plt.figure()
        ax_tmp = fig_tmp.add_subplot(111)
        cnt = ax_tmp.contour(smoothed.T, levels=[0.5])
        for path in cnt.collections[0].get_paths():
            verts = path.vertices
            one_contour = [(float(v[0]), float(v[1])) for v in verts]
            if len(one_contour) >= 12:
                contour_list.append(one_contour)
        plt.close(fig_tmp)

    return contour_list


def split_contour_on_jumps(contour, jump_thresh=35.0):
    if len(contour) < 2:
        return [contour]

    segments = []
    current = [contour[0]]
    for i in range(1, len(contour)):
        x0, y0 = contour[i - 1]
        x1, y1 = contour[i]
        if abs(x1 - x0) > jump_thresh or abs(y1 - y0) > jump_thresh:
            if len(current) > 1:
                segments.append(current)
            current = [contour[i]]
        else:
            current.append(contour[i])

    if len(current) > 1:
        segments.append(current)
    return segments


def save_csv(contours, filename):
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['contour_id', 'x', 'y'])
        for cid, contour in enumerate(contours):
            for x, y in contour:
                writer.writerow([cid, x, y])


# ============================================================
# 7. 可视化辅助
# ============================================================
def build_rgb(exist_data, mat_data):
    rgb = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    vac_mask = exist_data < 0.5
    mask_mask = (exist_data >= 0.5) & (mat_data == 2)
    si_mask = (exist_data >= 0.5) & (mat_data == 1)

    rgb[vac_mask] = to_rgb("#008CFF")
    rgb[mask_mask] = to_rgb("#20DDE0")
    rgb[si_mask] = to_rgb("#00008B")
    return rgb


def plot_state(ax, exist_data, mat_data, history_lines, current_count):
    ax.clear()
    rgb = build_rgb(exist_data, mat_data)
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='upper')

    n_hist = max(1, len(history_lines))
    for idx, contours_in_frame in enumerate(history_lines):
        alpha = 0.22 + 0.78 * (idx / n_hist)
        color = 'white' if idx == len(history_lines) - 1 else 'red'
        lw = 1.6 if idx == len(history_lines) - 1 else 0.7

        for contour in contours_in_frame:
            for seg in split_contour_on_jumps(contour, jump_thresh=35.0):
                xs = [p[0] for p in seg]
                ys = [p[1] for p in seg]
                ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw)

    first_left = LEFT_BORDER
    last_right = RIGHT_BORDER + (NUM_TRENCH - 1) * (CD + SPACE)
    x_min = max(0, first_left - VIEW_MARGIN_X)
    x_max = min(ROWS, last_right + VIEW_MARGIN_X)
    y_min = max(0, VACUUM_Y - VIEW_MARGIN_Y_TOP)
    y_max = min(COLS, MASK_BOTTOM_Y + VIEW_MARGIN_Y_BOTTOM)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, 0)
    ax.set_title(f"SEM-oriented Bowing Reproduction v2: {current_count}/{TOTAL_PARTICLES}")


# ============================================================
# 8. 主程序
# ============================================================
def main():
    if not HAS_SKIMAGE:
        print("Warning: 未检测到 scikit-image，将退回 Matplotlib contour 引擎。")
        print("建议安装: pip install scikit-image")

    init_grid()

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    history_lines = []

    print(">>> 模拟开始：Mask 脚部刻蚀 + 角点增强 + 深度衰减版 <<<")
    num_batches = TOTAL_PARTICLES // BATCH_SIZE

    for i in range(num_batches):
        simulate_batch()

        if ENABLE_WEAK_SMOOTH and (i % SMOOTH_EVERY == 0) and i > 0:
            smooth_grid()

        if i % DISPLAY_EVERY == 0:
            ti.sync()
            mat_data = grid_material.to_numpy()
            exist_data = grid_exist.to_numpy()

            contours = get_contours(exist_data)
            current_count = i * BATCH_SIZE
            save_csv(contours, f"contour_{current_count}.csv")

            if len(contours) > 0:
                history_lines.append(contours)
                if len(history_lines) > MAX_HISTORY:
                    history_lines.pop(0)

            plot_state(ax, exist_data, mat_data, history_lines, current_count)
            plt.pause(0.01)
            print(f"进度: {i / num_batches:.1%}", end='\r')

    ti.sync()
    final_exist = grid_exist.to_numpy()
    final_mat = grid_material.to_numpy()
    final_contours = get_contours(final_exist)
    save_csv(final_contours, "contour_final.csv")

    plot_state(ax, final_exist, final_mat, history_lines + [final_contours], TOTAL_PARTICLES)
    fig.savefig(os.path.join(SAVE_DIR, "final_profile.png"), dpi=200, bbox_inches='tight')

    print("\n>>> 模拟完成。已保存 contour_final.csv 和 final_profile.png")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()