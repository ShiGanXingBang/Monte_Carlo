import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter1d


# Literature-inspired single-opening bowing profile reconstruction.
# Goal: reproduce one "middle wider, top/bottom narrower" etched profile first.
#
# Mechanisms added in half_width_profile():
# 1. Stronger passivation near the mask opening suppresses top-side lateral etch.
# 2. Mid-depth bowing source represents wider ion angular spread and sidewall-reflected/scattered ions.
# 3. ARDE transport loss reduces lateral etch capability with depth.
# 4. Bottom re-passivation / redeposition narrows the profile again near the trench bottom.

SAVE_DIR = "bowing_single_output"
ROWS = 720
COLS = 520

VACUUM_TOP = 80
MASK_BOTTOM = 170
SURFACE_Y = 220

CENTER_X = ROWS // 2
TOP_OPENING = 84
ETCH_DEPTH = 250

MASK_TOP_OPENING = 62
MASK_BOTTOM_OPENING = TOP_OPENING
MASK_HALF_EXTRA = 120

HISTORY_STEPS = 8


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def half_width_profile(depth: np.ndarray, progress: float = 1.0) -> np.ndarray:
    """Return the trench half-width at each depth."""
    depth = depth.astype(np.float64)

    # Mechanism 1: stronger protection very close to the mask opening.
    top_passivation = 4.8 * np.exp(-((depth - 16.0) / 22.0) ** 2)

    # Mechanism 2: sidewall bowing peak at mid-depth.
    mid_bow = (9.0 + 11.0 * progress) * np.exp(-((depth - 112.0) / 72.0) ** 2)

    # Mechanism 3: ARDE-like transport loss with depth.
    arde_decay = np.exp(-depth / 310.0)

    # Mechanism 4: bottom narrowing due to re-passivation / redeposition.
    bottom_neck = (7.0 + 7.0 * progress) * sigmoid((depth - 196.0) / 18.0)

    baseline_open = 2.2 * (1.0 - np.exp(-depth / 70.0))

    half_width = (
        TOP_OPENING / 2.0
        + baseline_open
        + mid_bow * arde_decay
        - top_passivation
        - bottom_neck
    )

    toe_rounding = 2.2 * np.exp(-((depth - depth.max()) / 17.0) ** 2)
    half_width -= toe_rounding
    half_width = gaussian_filter1d(half_width, sigma=2.0, mode="nearest")

    min_half = TOP_OPENING / 2.0 - 4.0
    max_half = TOP_OPENING / 2.0 + 16.0
    return np.clip(half_width, min_half, max_half)

def mask_opening_half(y: int) -> float:
    if y < VACUUM_TOP or y > MASK_BOTTOM:
        return 0.0
    t = (y - VACUUM_TOP) / max(1, MASK_BOTTOM - VACUUM_TOP)
    opening = MASK_TOP_OPENING + (MASK_BOTTOM_OPENING - MASK_TOP_OPENING) * t
    return opening / 2.0


def build_material_grid(final_half_width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid_exist = np.ones((ROWS, COLS), dtype=np.float32)
    grid_material = np.ones((ROWS, COLS), dtype=np.int32)

    grid_exist[:, :VACUUM_TOP] = 0.0
    grid_material[:, :VACUUM_TOP] = 0

    for y in range(VACUUM_TOP, MASK_BOTTOM):
        opening_half = mask_opening_half(y)
        body_half = opening_half + MASK_HALF_EXTRA
        left_void = int(round(CENTER_X - opening_half))
        right_void = int(round(CENTER_X + opening_half))
        left_mask = int(round(CENTER_X - body_half))
        right_mask = int(round(CENTER_X + body_half))

        grid_material[:, y] = 2
        grid_exist[:, y] = 1.0
        grid_exist[max(0, left_void):min(ROWS, right_void), y] = 0.0
        grid_material[max(0, left_void):min(ROWS, right_void), y] = 0
        grid_material[:max(0, left_mask), y] = 2
        grid_material[min(ROWS, right_mask):, y] = 2

    # Open the channel from mask bottom to substrate surface.
    for y in range(MASK_BOTTOM, SURFACE_Y):
        left = int(round(CENTER_X - TOP_OPENING / 2.0))
        right = int(round(CENTER_X + TOP_OPENING / 2.0))
        grid_exist[max(0, left):min(ROWS, right), y] = 0.0
        grid_material[max(0, left):min(ROWS, right), y] = 0

    grid_material[:, SURFACE_Y:] = 1
    grid_exist[:, SURFACE_Y:] = 1.0

    for depth_idx, half in enumerate(final_half_width):
        y = SURFACE_Y + depth_idx
        if y >= COLS:
            break
        left = int(round(CENTER_X - half))
        right = int(round(CENTER_X + half))
        grid_exist[max(0, left):min(ROWS, right), y] = 0.0
        grid_material[max(0, left):min(ROWS, right), y] = 0

    return grid_exist, grid_material


def contour_from_profile(half_width: np.ndarray) -> np.ndarray:
    ys = np.arange(len(half_width), dtype=np.float64) + SURFACE_Y
    left = CENTER_X - half_width
    right = CENTER_X + half_width

    x = np.concatenate([left, right[::-1], left[:1]])
    y = np.concatenate([ys, ys[::-1], ys[:1]])
    return np.column_stack([x, y])


def save_contour_csv(points: np.ndarray, filename: str) -> None:
    ensure_dir(SAVE_DIR)
    path = os.path.join(SAVE_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(points.tolist())

def render(grid_exist: np.ndarray, grid_material: np.ndarray, history: list[np.ndarray], out_png: str) -> None:
    rgb = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    vac_mask = grid_exist < 0.5
    mask_mask = (grid_exist >= 0.5) & (grid_material == 2)
    si_mask = (grid_exist >= 0.5) & (grid_material == 1)

    rgb[vac_mask] = to_rgb("#1689e5")
    rgb[mask_mask] = to_rgb("#1fd8dd")
    rgb[si_mask] = to_rgb("#07078f")

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=160)
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="upper")

    for idx, contour in enumerate(history):
        color = "white" if idx == len(history) - 1 else "red"
        alpha = 0.22 + 0.78 * (idx + 1) / len(history)
        ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.15, alpha=alpha)

    ax.set_title("Single-opening bowing profile reconstruction")
    ax.set_xlim(260, 460)
    ax.set_ylim(SURFACE_Y + ETCH_DEPTH + 15, 40)
    ax.set_xlabel("x / pixel")
    ax.set_ylabel("y / pixel")
    fig.tight_layout()

    ensure_dir(SAVE_DIR)
    fig.savefig(os.path.join(SAVE_DIR, out_png), bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    ensure_dir(SAVE_DIR)

    depth = np.arange(ETCH_DEPTH, dtype=np.float64)
    history_profiles = []

    for step in range(1, HISTORY_STEPS + 1):
        progress = step / HISTORY_STEPS
        half = half_width_profile(depth, progress=progress)
        history_profiles.append(contour_from_profile(half))

    final_half = half_width_profile(depth, progress=1.0)
    grid_exist, grid_material = build_material_grid(final_half)

    save_contour_csv(history_profiles[-1], "single_bowing_contour.csv")
    render(grid_exist, grid_material, history_profiles, out_png="single_bowing_profile.png")

    width_report = np.column_stack([depth, final_half * 2.0])
    np.savetxt(
        os.path.join(SAVE_DIR, "single_bowing_width_profile.csv"),
        width_report,
        delimiter=",",
        header="depth_px,width_px",
        comments="",
    )

    print("Saved:")
    print(os.path.join(SAVE_DIR, "single_bowing_profile.png"))
    print(os.path.join(SAVE_DIR, "single_bowing_contour.csv"))
    print(os.path.join(SAVE_DIR, "single_bowing_width_profile.csv"))


if __name__ == "__main__":
    main()
