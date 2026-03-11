import taichi as ti
import math

# =========================================================
# 1. 初始化
# =========================================================
# 如果你的 GPU 后端不稳定，可改成 ti.cpu
ti.init(arch=ti.gpu)

# ------------------ 网格参数（2D 截面） -------------------
SIZE_X = 520
SIZE_Y = 640

VACUUM = 28
MASK_BOTTOM = 58
TRENCH_HALF_WIDTH = 18
MASK_TAPER_DEG = 2.0

# ------------------ 仿真规模 -----------------------------
TOTAL_PARTICLES = 2_000_000
BATCH_SIZE = 2000
BATCH_PER_FRAME = 6

# ------------------ 工艺“有效参数” -----------------------
# 这里是工程化有效参数，不是严格实验一一映射
HBR_FLOW = 200.0
O2_FLOW = 3.0

# 物种比例（工程近似）
ION_FRAC = 0.12
BASE_O_FRAC = 0.06
O_FRAC = BASE_O_FRAC * (O2_FLOW / 3.0)
if O_FRAC > 0.15:
    O_FRAC = 0.15

HBR_FRAC = 1.0 - ION_FRAC - O_FRAC
if HBR_FRAC < 0.1:
    HBR_FRAC = 0.1

# ------------------ 表面状态上限 -------------------------
MAX_BR = 4
MAX_PASS = 4

# ------------------ 反应强度（可调） ---------------------
ION_REFLECT_MASK_BOOST = 0.30
ION_PASS_REMOVE_BASE = 0.40
ION_SI_ETCH_BASE = 0.06
ION_SI_ETCH_BR_GAIN = 0.08

HBR_STICK_BASE = 0.52
HBR_CHEM_ETCH_BASE = 0.018

O_STICK_BASE = 0.30
O_SIDEWALL_GAIN = 0.35
O_TOP_BOTTOM_GAIN = 0.08

ETCH_AMOUNT_ION = 0.34
ETCH_AMOUNT_NEUTRAL = 0.24

# =========================================================
# 2. 数据场
# =========================================================
# grid_exist: 0 = 空, >0 = 材料剩余程度
grid_exist = ti.field(dtype=ti.f32, shape=(SIZE_X, SIZE_Y))
# grid_material: 0=真空, 1=Si, 2=Mask
grid_material = ti.field(dtype=ti.i32, shape=(SIZE_X, SIZE_Y))

# Br 覆盖度
grid_count_br = ti.field(dtype=ti.i32, shape=(SIZE_X, SIZE_Y))
# O 诱导钝化层
grid_pass_o = ti.field(dtype=ti.i32, shape=(SIZE_X, SIZE_Y))

# GUI 显示图像
display_img = ti.Vector.field(3, dtype=ti.f32, shape=(SIZE_X, SIZE_Y))

# =========================================================
# 3. 物理辅助函数（2D）
# =========================================================
@ti.func
def clamp_int(v, lo, hi):
    return ti.max(lo, ti.min(hi, v))


@ti.func
def is_surface(x: int, y: int) -> ti.i32:
    # 注意：Taichi func 里不要在动态 if 中提前 return
    s = 0

    if grid_exist[x, y] >= 0.5:
        if x == 0 or x == SIZE_X - 1 or y == 0 or y == SIZE_Y - 1:
            s = 1
        else:
            if grid_exist[x + 1, y] < 0.5:
                s = 1
            if grid_exist[x - 1, y] < 0.5:
                s = 1
            if grid_exist[x, y + 1] < 0.5:
                s = 1
            if grid_exist[x, y - 1] < 0.5:
                s = 1

    return s


@ti.func
def get_surface_normal_2d(px: int, py: int):
    # 指向真空方向的法线
    nx, ny = 0.0, 0.0

    for i, j in ti.static(ti.ndrange((-1, 2), (-1, 2))):
        xx = px + i
        yy = py + j
        if 0 <= xx < SIZE_X and 0 <= yy < SIZE_Y:
            if grid_exist[xx, yy] < 0.5:
                nx += float(i)
                ny += float(j)

    norm = ti.sqrt(nx * nx + ny * ny) + 1e-6
    return nx / norm, ny / norm


@ti.func
def diffuse_reflect_2d(nx: float, ny: float):
    # 在法线半空间附近取一个随机方向
    base = ti.atan2(ny, nx)
    ang = base + (ti.random() - 0.5) * math.pi
    vx = ti.cos(ang)
    vy = ti.sin(ang)
    return vx, vy


@ti.func
def get_reflection_vector_2d(vx: float, vy: float, nx: float, ny: float, is_ion: int):
    rvx, rvy = 0.0, 0.0
    if is_ion == 1:
        # 镜面反射
        dot = vx * nx + vy * ny
        rvx = vx - 2.0 * dot * nx
        rvy = vy - 2.0 * dot * ny
    else:
        # 漫反射
        rvx, rvy = diffuse_reflect_2d(nx, ny)

    # 确保反射后朝向真空半空间
    if rvx * nx + rvy * ny < 0:
        rvx = -rvx
        rvy = -rvy

    norm = ti.sqrt(rvx * rvx + rvy * rvy) + 1e-6
    return rvx / norm, rvy / norm


@ti.func
def clear_cell_if_removed(x: int, y: int):
    if grid_exist[x, y] <= 0.0:
        grid_exist[x, y] = 0.0
        grid_material[x, y] = 0
        grid_count_br[x, y] = 0
        grid_pass_o[x, y] = 0

# =========================================================
# 4. 几何初始化（2D line trench）
# =========================================================
@ti.kernel
def init_grid():
    cx = SIZE_X / 2.0
    taper = ti.tan(MASK_TAPER_DEG * math.pi / 180.0)

    for i, j in grid_exist:
        grid_count_br[i, j] = 0
        grid_pass_o[i, j] = 0

        if j < VACUUM:
            grid_exist[i, j] = 0.0
            grid_material[i, j] = 0
        elif VACUUM <= j < MASK_BOTTOM:
            # 掩膜开口
            bias = (MASK_BOTTOM - j) * taper
            current_half_width = TRENCH_HALF_WIDTH + bias

            if ti.abs(i - cx) < current_half_width:
                grid_exist[i, j] = 0.0
                grid_material[i, j] = 0
            else:
                grid_exist[i, j] = 1.0
                grid_material[i, j] = 2
        else:
            grid_exist[i, j] = 1.0
            grid_material[i, j] = 1

# =========================================================
# 5. 核心仿真（2D）
# =========================================================
@ti.kernel
def simulate_batch():
    # 串行，降低多个粒子同时修改同一单元导致的随机冲突
    ti.loop_config(serialize=True)

    for _ in range(BATCH_SIZE):
        # ------------------ 发射位置 ------------------
        px = ti.random() * (SIZE_X - 1)
        py = 1.0

        # ------------------ 粒子类型 ------------------
        # 0 = HBr 中性粒子
        # 1 = 离子
        # 2 = O 中性粒子
        r = ti.random()
        species = 0
        if r < ION_FRAC:
            species = 1
        elif r < ION_FRAC + O_FRAC:
            species = 2
        else:
            species = 0

        # ------------------ 初始角分布 ----------------
        # y 向下为主入射方向
        sigma = 0.0
        if species == 1:
            sigma = 3.0 * math.pi / 180.0
        elif species == 2:
            sigma = 25.0 * math.pi / 180.0
        else:
            sigma = 20.0 * math.pi / 180.0

        theta = ti.abs(ti.randn() * sigma)
        theta = ti.min(theta, math.pi / 2.0 - 0.05)

        sign = 1.0
        if ti.random() < 0.5:
            sign = -1.0

        vx = sign * ti.sin(theta)
        vy = ti.cos(theta)

        alive = True
        steps = 0
        energy = 1.0
        ref_count = 0

        while alive and steps < 2000:
            steps += 1

            step_size = 0.55
            px_n = px + vx * step_size
            py_n = py + vy * step_size

            # x 周期边界
            if px_n < 0:
                px_n += SIZE_X
            if px_n >= SIZE_X:
                px_n -= SIZE_X

            # y 出界则粒子结束
            if py_n < 0 or py_n >= SIZE_Y:
                alive = False
                break

            ipx = int(px_n)
            ipy = int(py_n)

            if grid_exist[ipx, ipy] > 0.5:
                mat = grid_material[ipx, ipy]
                br_n = grid_count_br[ipx, ipy]
                pass_n = grid_pass_o[ipx, ipy]

                nx, ny = get_surface_normal_2d(ipx, ipy)

                # 入射角余弦（相对于表面法线）
                cos_theta = -(vx * nx + vy * ny)
                cos_theta = ti.max(0.0, ti.min(1.0, cos_theta))
                theta_coll = ti.acos(cos_theta)

                did_reflect = False

                # ======================================
                # A. 离子
                # ======================================
                if species == 1:
                    threshold = math.pi / 3.0
                    angle_else = ti.max(0.0, (theta_coll - threshold) / (math.pi / 2.0 - threshold))
                    prob_reflect = ti.min(1.0, angle_else)

                    if mat == 2:
                        prob_reflect = ti.min(1.0, prob_reflect + ION_REFLECT_MASK_BOOST)

                    if ti.random() < prob_reflect:
                        did_reflect = True
                        ref_count += 1

                    if did_reflect:
                        vx, vy = get_reflection_vector_2d(vx, vy, nx, ny, 1)
                        px = px + nx * 1.2
                        py = py + ny * 1.2

                        if mat == 2:
                            energy *= 0.90
                        elif pass_n > 0:
                            energy *= 0.75
                        else:
                            energy *= 0.55
                    else:
                        # 先去钝化
                        if pass_n > 0:
                            p_remove = ION_PASS_REMOVE_BASE + 0.25 * cos_theta + 0.18 * energy
                            p_remove = ti.min(0.95, p_remove)

                            if ti.random() < p_remove:
                                grid_pass_o[ipx, ipy] = clamp_int(grid_pass_o[ipx, ipy] - 1, 0, MAX_PASS)
                            alive = False

                        else:
                            if mat == 1:
                                # 离子辅助 Si 刻蚀
                                p_etch = ION_SI_ETCH_BASE
                                if br_n >= 1:
                                    p_etch += ION_SI_ETCH_BR_GAIN
                                if br_n >= 2:
                                    p_etch += ION_SI_ETCH_BR_GAIN
                                if br_n >= 3:
                                    p_etch += 0.05

                                p_etch *= (0.35 + 0.65 * cos_theta) * energy

                                if ti.random() < p_etch:
                                    grid_exist[ipx, ipy] -= ETCH_AMOUNT_ION
                                    clear_cell_if_removed(ipx, ipy)
                                alive = False

                            elif mat == 2:
                                # 掩膜轻微溅射
                                if ti.random() < 0.006 * energy:
                                    grid_exist[ipx, ipy] -= 0.12
                                    clear_cell_if_removed(ipx, ipy)
                                alive = False
                            else:
                                alive = False

                # ======================================
                # B. HBr 中性粒子
                # ======================================
                elif species == 0:
                    prob_reflect = 0.58 + 0.06 * br_n + 0.05 * pass_n
                    prob_reflect = ti.min(0.92, prob_reflect)

                    if ti.random() < prob_reflect:
                        did_reflect = True
                        ref_count += 1

                    if did_reflect:
                        vx, vy = get_reflection_vector_2d(vx, vy, nx, ny, 0)
                        px = px + nx * 1.0
                        py = py + ny * 1.0
                    else:
                        if mat == 1:
                            # Br 吸附
                            p_ads = HBR_STICK_BASE * (1.0 - 0.22 * pass_n)
                            p_ads = ti.max(0.02, p_ads)

                            if br_n < MAX_BR and ti.random() < p_ads:
                                grid_count_br[ipx, ipy] = clamp_int(grid_count_br[ipx, ipy] + 1, 0, MAX_BR)

                            # 化学刻蚀
                            p_chem = 0.0
                            if br_n >= 2 and pass_n == 0:
                                p_chem = HBR_CHEM_ETCH_BASE
                                if br_n >= 3:
                                    p_chem += 0.01

                                if ti.random() < p_chem:
                                    grid_exist[ipx, ipy] -= ETCH_AMOUNT_NEUTRAL
                                    clear_cell_if_removed(ipx, ipy)

                        alive = False

                # ======================================
                # C. O 中性粒子
                # ======================================
                else:
                    prob_reflect = 0.32 + 0.08 * br_n
                    prob_reflect = ti.min(0.75, prob_reflect)

                    if ti.random() < prob_reflect:
                        did_reflect = True
                        ref_count += 1

                    if did_reflect:
                        vx, vy = get_reflection_vector_2d(vx, vy, nx, ny, 0)
                        px = px + nx * 0.9
                        py = py + ny * 0.9
                    else:
                        if mat == 1:
                            # 侧壁更容易形成钝化
                            sidewall_factor = ti.abs(nx)
                            top_bottom_factor = ti.abs(ny)

                            p_pass = O_STICK_BASE * (O2_FLOW / 3.0)
                            p_pass += O_SIDEWALL_GAIN * sidewall_factor * (O2_FLOW / 3.0)
                            p_pass += O_TOP_BOTTOM_GAIN * top_bottom_factor * 0.5 * (O2_FLOW / 3.0)

                            # 饱和效应
                            p_pass *= (1.0 - 0.18 * pass_n)
                            p_pass = ti.max(0.02, ti.min(0.95, p_pass))

                            if pass_n < MAX_PASS and ti.random() < p_pass:
                                grid_pass_o[ipx, ipy] = clamp_int(grid_pass_o[ipx, ipy] + 1, 0, MAX_PASS)

                                # O 会抑制部分 Br 活性
                                if grid_count_br[ipx, ipy] > 0 and ti.random() < 0.45:
                                    grid_count_br[ipx, ipy] = clamp_int(grid_count_br[ipx, ipy] - 1, 0, MAX_BR)

                        alive = False
            else:
                px, py = px_n, py_n

# =========================================================
# 6. 表面状态松弛
# =========================================================
@ti.kernel
def relax_non_surface_states():
    for i, j in grid_exist:
        if grid_exist[i, j] < 0.5:
            grid_count_br[i, j] = 0
            grid_pass_o[i, j] = 0
        else:
            if is_surface(i, j) == 0:
                if grid_count_br[i, j] > 0:
                    grid_count_br[i, j] -= 1
                if grid_pass_o[i, j] > 0:
                    grid_pass_o[i, j] -= 1

# =========================================================
# 7. 2D 显示
# =========================================================
@ti.kernel
def update_display():
    for i, j in display_img:
        # 先给默认值，避免 Taichi 作用域问题
        c = ti.Vector([1.0, 1.0, 1.0])

        if grid_exist[i, j] < 0.5:
            c = ti.Vector([1.0, 1.0, 1.0])
        else:
            mat = grid_material[i, j]
            br_n = grid_count_br[i, j]
            pass_n = grid_pass_o[i, j]

            if mat == 2:
                # 掩膜：粉色
                c = ti.Vector([0.95, 0.82, 0.86])
            else:
                # Si：浅蓝灰
                c = ti.Vector([0.72, 0.78, 0.92])

                # Br 覆盖：偏绿色
                if br_n > 0:
                    alpha_br = 0.10 * br_n
                    if alpha_br > 0.30:
                        alpha_br = 0.30
                    c = c * (1.0 - alpha_br) + ti.Vector([0.45, 0.82, 0.42]) * alpha_br

                # O 钝化：偏橙色
                if pass_n > 0:
                    alpha_o = 0.14 * pass_n
                    if alpha_o > 0.50:
                        alpha_o = 0.50
                    c = c * (1.0 - alpha_o) + ti.Vector([0.95, 0.62, 0.18]) * alpha_o

        display_img[i, j] = c

# =========================================================
# 8. 主程序
# =========================================================
def main():
    print(">>> 初始化 2D HBr/O2 刻蚀模型 ...")
    print(f">>> HBR_FLOW = {HBR_FLOW}")
    print(f">>> O2_FLOW  = {O2_FLOW}")
    print(">>> 模型机制：Br 覆盖 + O 侧壁钝化 + 离子去钝化/辅助刻蚀")
    print(">>> 这是工程化轮廓趋势模拟，不是严格实验标定版。")

    init_grid()

    gui = ti.GUI("2D HBr/O2 Etching Profile", res=(SIZE_X, SIZE_Y))
    step_count = 0
    sim_finished = False
    frame_id = 0

    while gui.running:
        if not sim_finished:
            for _ in range(BATCH_PER_FRAME):
                if step_count < TOTAL_PARTICLES:
                    simulate_batch()
                    step_count += BATCH_SIZE
                else:
                    sim_finished = True
                    break

            if frame_id % 5 == 0:
                relax_non_surface_states()

        update_display()
        gui.set_image(display_img)

        progress = min(1.0, step_count / TOTAL_PARTICLES)
        gui.text(content=f"Particles: {step_count}/{TOTAL_PARTICLES}", pos=(0.02, 0.97), color=0x111111)
        gui.text(content=f"Progress: {progress * 100:.1f}%", pos=(0.02, 0.94), color=0x111111)
        gui.text(content=f"O2_FLOW = {O2_FLOW}", pos=(0.02, 0.91), color=0x111111)
        gui.text(content="ESC to exit", pos=(0.02, 0.88), color=0x111111)

        gui.show()
        frame_id += 1


if __name__ == "__main__":
    main()