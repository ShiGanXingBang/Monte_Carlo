import csv
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Taichi Monte Carlo trench-bowing model.
#
# Retained framework:
# - Taichi particle tracing
# - reflection-first collision handling
# - ion / neutral split
# - contour export
#
# Added bowing mechanisms:
# 1. Reflected ions do not etch sidewalls immediately; they deposit a secondary sidewall attack flux.
# 2. That attack flux is strongest at mid depth, weaker near the trench entrance and bottom.
# 3. Direct ion sidewall etch is intentionally weak so the trench does not simply taper inward.
# 4. Bottom etch remains active, but is reduced at large depth to keep the bottom narrower than the middle.

try:
    ti.init(arch=ti.gpu, device_memory_fraction=0.7, offline_cache=False)
except Exception:
    ti.init(arch=ti.cpu, offline_cache=False)

SAVE_DIR = r"Csv\SingleBowing_Taichi"
os.makedirs(SAVE_DIR, exist_ok=True)

MAT_VACUUM = 0
MAT_SI = 1
MAT_MASK = 2

ROWS, COLS = 640, 560
VACUUM = 120
MASK_BOTTOM = 220
LEFT_BORDER = 270
RIGHT_BORDER = 350
TRENCH_DEPTH_TARGET = 230
MASK_ANGLE_DEG = 12

TOTAL_PARTICLES = 4_000_000
BATCH_SIZE = 4000
SYNC_INTERVAL = 40
ION_FRACTION = 0.22

ION_ANGLE_SIGMA = 1.8 * (math.pi / 180.0)
NEUTRAL_ANGLE_SIGMA = 7.2 * (math.pi / 180.0)
ENERGY_INITIAL = 1.0
ENERGY_MIN = 0.06
ENERGY_LOSS_MASK_REFLECTION = 0.94
ENERGY_LOSS_SI_REFLECTION = 0.72
CL_SATURATION = 3

grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))
grid_count_cl = ti.field(dtype=ti.i32, shape=(ROWS, COLS))
grid_temp = ti.field(dtype=ti.f32, shape=(ROWS, COLS))
sidewall_flux = ti.field(dtype=ti.f32, shape=(ROWS, COLS))


@ti.func
def clamp01(x: float) -> float:
    return ti.max(0.0, ti.min(1.0, x))


@ti.func
def depth_norm(py: int) -> float:
    return clamp01((py - MASK_BOTTOM) / float(TRENCH_DEPTH_TARGET))


@ti.func
def gaussian_term(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return ti.exp(-z * z)


@ti.func
def logistic_term(x: float, center: float, sharpness: float) -> float:
    return 1.0 / (1.0 + ti.exp(-(x - center) / sharpness))


@ti.func
def sidewall_depth_gate(py: int) -> float:
    d = depth_norm(py)
    mid = gaussian_term(d, 0.42, 0.16)
    top_supp = 1.0 - 0.85 * gaussian_term(d, 0.05, 0.08)
    bottom_supp = 1.0 - 0.82 * logistic_term(d, 0.80, 0.05)
    return clamp01(mid * top_supp * bottom_supp)


@ti.func
def get_surface_normal(px: int, py: int):
    nx, ny = 0.0, 0.0
    for i, j in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        if 0 <= px + i < ROWS and 0 <= py + j < COLS:
            if grid_exist[px + i, py + j] < 0.5:
                nx += float(i)
                ny += float(j)
    norm = ti.sqrt(nx * nx + ny * ny) + 1e-6
    return nx / norm, ny / norm


@ti.func
def get_reflection_vector(vx: float, vy: float, nx: float, ny: float, is_ion: int):
    rvx, rvy = 0.0, 0.0
    if is_ion == 1:
        dot = vx * nx + vy * ny
        rvx = vx - 2.0 * dot * nx
        rvy = vy - 2.0 * dot * ny
    else:
        tx, ty = -ny, nx
        sin_theta = (ti.random() - 0.5) * 2.0
        cos_theta = ti.sqrt(ti.max(0.0, 1.0 - sin_theta * sin_theta))
        rvx = nx * cos_theta + tx * sin_theta
        rvy = ny * cos_theta + ty * sin_theta
    norm = ti.sqrt(rvx * rvx + rvy * rvy) + 1e-6
    return rvx / norm, rvy / norm


@ti.func
def reflection_probability(mat: int, theta_coll: float, is_ion: int, cl_cov: int, py: int, nx: float) -> float:
    d_gate = sidewall_depth_gate(py)
    sidewallness = ti.abs(nx)
    prob = 0.0
    if is_ion == 1:
        threshold = math.pi / 3.1
        prob = clamp01((theta_coll - threshold) / (math.pi / 2 - threshold + 1e-6))
        if mat == MAT_MASK:
            prob = ti.max(0.85, prob + 0.15)
        else:
            prob = ti.min(0.96, prob + 0.35 * d_gate * sidewallness)
    else:
        prob = 0.42 + 0.08 * cl_cov
        if mat == MAT_MASK:
            prob += 0.08
        prob = ti.min(0.95, prob)
    return prob


@ti.func
def update_ion_energy(current_energy: float, mat_of_surface: int) -> float:
    new_energy = current_energy
    if mat_of_surface == MAT_MASK:
        new_energy = current_energy * ENERGY_LOSS_MASK_REFLECTION
    elif mat_of_surface == MAT_SI:
        new_energy = current_energy * ENERGY_LOSS_SI_REFLECTION
    return ti.max(ENERGY_MIN, new_energy)


@ti.func
def direct_ion_etch_probability(mat: int, cl_cov: int, energy: float, theta_coll: float,
                                py: int, nx: float, ny: float) -> float:
    prob = 0.02 * ti.sqrt(energy)
    sidewallness = ti.abs(nx)
    bottomness = ti.abs(ny)
    d = depth_norm(py)

    if mat != MAT_MASK:
        prob = 0.12
    if cl_cov > 0:
        prob += 0.16
    prob *= ti.sqrt(energy)
    prob *= 0.72 + 0.28 * ti.max(0.15, ti.cos(theta_coll))

    if sidewallness > 0.45:
        # Keep direct sidewall etch weak. Bowing should come from reflected-ion flux.
        prob *= 0.18
    else:
        # Bottom etch remains active but is reduced deep in the trench.
        if d > 0.78:
            prob *= 0.72
        if bottomness > 0.65 and d > 0.85:
            prob *= 0.62

    return ti.min(0.8, ti.max(0.0, prob))


@ti.kernel
def smooth_grid():
    w_center = 0.988
    w_neighbor = (1.0 - w_center) / 4.0
    for i, j in grid_exist:
        if 1 <= i < ROWS - 1 and 1 <= j < COLS - 1:
            val = (grid_exist[i, j] * w_center +
                   (grid_exist[i + 1, j] + grid_exist[i - 1, j] +
                    grid_exist[i, j + 1] + grid_exist[i, j - 1]) * w_neighbor)
            grid_temp[i, j] = val
        else:
            grid_temp[i, j] = grid_exist[i, j]
    for i, j in grid_exist:
        grid_exist[i, j] = grid_temp[i, j]


@ti.kernel
def init_grid():
    angle_rad = MASK_ANGLE_DEG * math.pi / 180.0
    k_mask = ti.abs(ti.tan(angle_rad))
    for i, j in grid_exist:
        grid_count_cl[i, j] = 0
        sidewall_flux[i, j] = 0.0
        if j <= VACUUM:
            grid_exist[i, j] = 0.0
            grid_material[i, j] = MAT_VACUUM
        elif j < MASK_BOTTOM:
            offset = int((MASK_BOTTOM - j) * k_mask)
            left_cur = max(0, min(LEFT_BORDER - offset, ROWS - 1))
            right_cur = max(0, min(RIGHT_BORDER + offset, ROWS - 1))
            if left_cur < i < right_cur:
                grid_exist[i, j] = 0.0
                grid_material[i, j] = MAT_VACUUM
            else:
                grid_exist[i, j] = 1.0
                grid_material[i, j] = MAT_MASK
        else:
            grid_exist[i, j] = 1.0
            grid_material[i, j] = MAT_SI


@ti.kernel
def apply_sidewall_flux():
    for i, j in grid_exist:
        if grid_material[i, j] == MAT_SI and grid_exist[i, j] > 0.5:
            flux = sidewall_flux[i, j]
            if flux > 0.0:
                etch_prob = ti.min(0.78, flux * 1.8)
                if ti.random() < etch_prob:
                    grid_exist[i, j] = ti.max(0.0, grid_exist[i, j] - 0.30)
                    if grid_exist[i, j] <= 0.0:
                        grid_material[i, j] = MAT_VACUUM
                sidewall_flux[i, j] *= 0.20


@ti.kernel
def simulate_batch():
    for _ in range(BATCH_SIZE):
        px, py = ti.random() * (ROWS - 1), 1.0
        is_ion = 0
        if ti.random() < ION_FRACTION:
            is_ion = 1

        sigma = ION_ANGLE_SIGMA if is_ion == 1 else NEUTRAL_ANGLE_SIGMA
        angle = ti.randn() * sigma
        angle = ti.max(-math.pi / 2, ti.min(math.pi / 2, angle))
        vx, vy = ti.sin(angle), ti.cos(angle)

        energy = ENERGY_INITIAL
        alive = True
        steps = 0
        ref_count = 0

        while alive and steps < 3200:
            steps += 1
            px_n, py_n = px + vx * 1.08, py + vy * 1.08
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
                nx, ny = get_surface_normal(ipx, ipy)
                cos_theta = clamp01(-(vx * nx + vy * ny))
                theta_coll = ti.acos(cos_theta)
                sidewallness = ti.abs(nx)

                prob_reflect = reflection_probability(mat, theta_coll, is_ion, cl_n, ipy, nx)
                did_reflect = False
                if ti.random() < prob_reflect:
                    did_reflect = True
                    ref_count += 1
                    if is_ion == 1:
                        energy = update_ion_energy(energy, mat)

                if did_reflect:
                    if is_ion == 1 and mat == MAT_SI and sidewallness > 0.45:
                        gate = sidewall_depth_gate(ipy)
                        if gate > 0.01:
                            flux_add = 0.34 * energy * gate * (0.45 + 0.55 * sidewallness)
                            sidewall_flux[ipx, ipy] += flux_add
                            bx = ipx - int(nx)
                            if 0 <= bx < ROWS and grid_material[bx, ipy] == MAT_SI:
                                sidewall_flux[bx, ipy] += flux_add * 0.90
                    vx, vy = get_reflection_vector(vx, vy, nx, ny, is_ion)
                    px = px + nx * 1.4
                    py = py + ny * 1.4
                else:
                    if is_ion == 1:
                        prob_etch = direct_ion_etch_probability(mat, cl_n, energy, theta_coll, ipy, nx, ny)
                        if ti.random() < prob_etch:
                            grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - 0.22)
                            if grid_exist[ipx, ipy] <= 0.0:
                                grid_material[ipx, ipy] = MAT_VACUUM
                        alive = False
                    else:
                        prob_etch = 0.0
                        if mat == MAT_SI and cl_n >= CL_SATURATION:
                            prob_etch = 0.03
                        if ti.random() < prob_etch:
                            grid_exist[ipx, ipy] = ti.max(0.0, grid_exist[ipx, ipy] - 0.18)
                            if grid_exist[ipx, ipy] <= 0.0:
                                grid_material[ipx, ipy] = MAT_VACUUM
                                grid_count_cl[ipx, ipy] = 0
                        else:
                            if mat == MAT_SI and cl_n < CL_SATURATION and ti.random() < (1.0 - cl_n / 4.0) * 0.32:
                                grid_count_cl[ipx, ipy] += 1
                        alive = False
            else:
                px, py = px_n, py_n


def get_contour_points(raw_grid):
    smoothed = gaussian_filter(raw_grid.astype(float), sigma=1.0)
    if HAS_SKIMAGE:
        contours = measure.find_contours(smoothed, 0.5)
        if not contours:
            return []
        # Keep the contour around the central trench instead of the outer boundaries.
        center_x = (LEFT_BORDER + RIGHT_BORDER) / 2.0
        best = None
        best_score = 1e18
        for contour in contours:
            xs = contour[:, 1]
            ys = contour[:, 0]
            if ys.max() < MASK_BOTTOM - 5:
                continue
            score = abs(xs.mean() - center_x) - 0.001 * len(contour)
            if score < best_score:
                best_score = score
                best = contour
        if best is None:
            return []
        return [(p[1], p[0]) for p in best]
    return []


def save_csv(points, filename):
    with open(os.path.join(SAVE_DIR, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(points)


def save_snapshot(name: str):
    mat_data = grid_material.to_numpy()
    exist_data = grid_exist.to_numpy()
    rgb = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    rgb[exist_data < 0.5] = to_rgb('#008CFF')
    rgb[(exist_data >= 0.5) & (mat_data == MAT_MASK)] = to_rgb('#00FFFF')
    rgb[(exist_data >= 0.5) & (mat_data == MAT_SI)] = to_rgb('#00008B')

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='upper')
    ax.set_xlim(170, 450)
    ax.set_ylim(MASK_BOTTOM + TRENCH_DEPTH_TARGET + 60, 60)
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f'{name}.png'), bbox_inches='tight')
    plt.close(fig)


def main():
    init_grid()
    history_lines = []
    num_batches = TOTAL_PARTICLES // BATCH_SIZE
    start = time.time()

    for i in range(num_batches):
        simulate_batch()
        apply_sidewall_flux()
        smooth_grid()
        if i % SYNC_INTERVAL == 0:
            ti.sync()
            points = get_contour_points(grid_exist.to_numpy())
            current_count = i * BATCH_SIZE
            save_csv(points, f'contour_{current_count}.csv')
            if points:
                history_lines.append(points)
            print(f'progress: {i / num_batches:.1%} elapsed: {time.time() - start:.0f}s', end='\r')

    ti.sync()
    final_points = get_contour_points(grid_exist.to_numpy())
    save_csv(final_points, 'contour_final.csv')
    if final_points:
        history_lines.append(final_points)
    save_snapshot('single_bowing_taichi_final')
    print(f"\nfinished in {time.time() - start:.1f}s")


if __name__ == '__main__':
    main()
