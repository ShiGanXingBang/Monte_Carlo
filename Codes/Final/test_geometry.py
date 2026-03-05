"""测试五槽几何结构"""
import taichi as ti
import math
import numpy as np
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

# 常量定义
GRID_SCALE_NM = 0.2
TRENCH_WIDTHS = [160, 140, 120, 100, 80]
PILLAR_WIDTH = 100
SIDE_MASK_WIDTH = 400
TAPER_ANGLE_DEG = 15
vacuum = 100
deep_border = 230

# 计算所需网格宽度
total_width_nm = (SIDE_MASK_WIDTH * 2 +
                  sum(TRENCH_WIDTHS) +
                  PILLAR_WIDTH * (len(TRENCH_WIDTHS) - 1))
ROWS = int(total_width_nm / GRID_SCALE_NM) + 50
COLS = 700

print(f'计算得到的网格宽度: {ROWS}')
print(f'总宽度: {total_width_nm} nm')

# Taichi 数据场
grid_exist = ti.field(dtype=ti.f32, shape=(ROWS, COLS))
grid_material = ti.field(dtype=ti.i32, shape=(ROWS, COLS))

# 初始化
@ti.kernel
def init_grid():
    angle_rad = TAPER_ANGLE_DEG * math.pi / 180.0
    k_mask = ti.abs(ti.tan(angle_rad))

    for i, j in grid_exist:
        if j <= vacuum:
            grid_exist[i, j] = 0.0
            grid_material[i, j] = 0

        current_x = int(SIDE_MASK_WIDTH / GRID_SCALE_NM)

        if j < deep_border and j > vacuum:
            if i < current_x:
                grid_material[i, j] = 2
                grid_exist[i, j] = 1.0

        for n in range(len(TRENCH_WIDTHS)):
            trench_width = TRENCH_WIDTHS[n]
            trench_width_grid = int(trench_width / GRID_SCALE_NM)
            pillar_width_grid = int(PILLAR_WIDTH / GRID_SCALE_NM)

            trench_left = current_x
            trench_right = current_x + trench_width_grid

            for y in range(vacuum, deep_border):
                offset = int((deep_border - y) * k_mask)
                l_cur = max(0, min(trench_left - offset, ROWS - 1))
                r_cur = max(0, min(trench_right + offset, ROWS - 1))

                l_side = max(0, min(trench_left - int(pillar_width_grid / 2), ROWS - 1))
                r_side = max(0, min(trench_right + int(pillar_width_grid / 2), ROWS - 1))

                for x in range(l_side, r_side):
                    if not (l_cur < x < r_cur):
                        grid_material[x, y] = 2
                        grid_exist[x, y] = 1.0
                    else:
                        grid_material[x, y] = 0
                        grid_exist[x, y] = 0.0

            current_x = trench_right + pillar_width_grid

        if j < deep_border and j > vacuum:
            if i >= current_x:
                grid_material[i, j] = 2
                grid_exist[i, j] = 1.0

        if j >= deep_border and j < COLS:
            if grid_exist[i, j] == 0.0:
                grid_exist[i, j] = 1.0
                grid_material[i, j] = 1

init_grid()

# 验证几何结构
print()
print('几何结构验证:')

current_x = int(SIDE_MASK_WIDTH / GRID_SCALE_NM)
print(f'左侧掩膜结束位置: x={current_x} ({current_x * GRID_SCALE_NM:.0f} nm)')

for n, trench_nm in enumerate(TRENCH_WIDTHS):
    trench_width_grid = int(trench_nm / GRID_SCALE_NM)
    pillar_width_grid = int(PILLAR_WIDTH / GRID_SCALE_NM)

    trench_left = current_x
    trench_right = current_x + trench_width_grid

    print(f'槽 {n+1}: {trench_nm} nm ({trench_width_grid} px) | x=[{trench_left}, {trench_right}]')

    current_x = trench_right + pillar_width_grid

print(f'右侧掩膜起始位置: x={current_x} ({current_x * GRID_SCALE_NM:.0f} nm)')
print()

# 可视化几何结构
mat_data = grid_material.to_numpy()
exist_data = grid_exist.to_numpy()

rgb = np.zeros((ROWS, COLS, 3), dtype=np.float32)
vac_mask = (exist_data < 0.5)
mask_mask = (exist_data >= 0.5) & (mat_data == 2)
si_mask = (exist_data >= 0.5) & (mat_data == 1)

rgb[vac_mask] = to_rgb("#008CFF")   # 真空：蓝色
rgb[mask_mask] = to_rgb("#00FFFF")  # 掩膜：青色
rgb[si_mask] = to_rgb("#00008B")    # Si：深蓝色

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='upper')
ax.set_title(f'五槽几何结构 (总宽度: {total_width_nm} nm)')
ax.set_xlabel('x (grid)')
ax.set_ylabel('y (grid)')
plt.tight_layout()
plt.savefig('test_geometry.png', dpi=150)
print('几何结构图像已保存至: test_geometry.png')
print('初始化测试完成！')
