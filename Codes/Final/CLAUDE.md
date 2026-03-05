# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

本项目是一个基于 Monte Carlo 方法的等离子体刻蚀（Plasma Etching）2D 仿真，使用 Taichi 进行 GPU 加速计算，模拟离子和中性粒子（Cl 自由基）对半导体衬底（Si）通过光刻掩膜进行刻蚀的过程。

## 运行方法

```bash
python Reaction_0209.py
# 或
python Reaction_0208.py
python Restore_Reaction_226.py
```

依赖库：`taichi`, `numpy`, `matplotlib`, `scipy`, `scikit-image`（可选，无则退化为 matplotlib 轮廓引擎）

输出 CSV 文件保存至 `Csv\Test_MarchingSquares_2026\`（运行时自动创建）。

## 三个文件的关键差异

| 文件 | `RATIO`（中性/总粒子比） | `vacuum` | `TOTAL_PARTICLES` | 反射/刻蚀逻辑特点 |
|---|---|---|---|---|
| `Reaction_0208.py` | 20/21（中性为主） | 150 | 4000万 | 反射后离子刻蚀概率×0.5 |
| `Reaction_0209.py` | 2/21（离子为主） | 150 | 4000万 | 新增 `energy` 变量追踪粒子能量衰减；掩膜反射保底 0.8 |
| `Restore_Reaction_226.py` | 10/11（中性为主） | 100 | 2000万 | 较早期参数，掩膜反射概率仅+0.2 |

## 代码架构（六个固定模块）

所有文件共享相同结构：

**1. 环境初始化**：`ti.init(arch=ti.gpu)`，定义全局几何常量（`ROWS=1000, COLS=700`，掩膜位置/尺寸/数量）和 Taichi 数据场。

**2. 物理辅助函数（`@ti.func`）**：
- `get_surface_normal(px, py)`：在 5×5 邻域内通过真空单元位置计算表面法线
- `get_reflection_vector(...)`：离子用镜面反射，中性粒子用 Lambertian 漫反射
- `smooth_grid()`：每 batch 后对 `grid_exist` 做加权平均平滑（w_center=0.98）

**3. 几何初始化（`init_grid()`）**：在 `grid_exist` 和 `grid_material` 中构建多槽掩膜结构（`Num=3` 个槽），掩膜侧壁带 15° 倾角。

**4. 核心仿真（`simulate_batch()`，`@ti.kernel`）**：每批次并行模拟 `BATCH_SIZE=4000` 个粒子，逻辑为"反射优先"：
- 粒子从顶部（`py=1`）以高斯角分布射入
- 碰撞时先判断是否反射，若反射则更新速度方向并弹开
- 未反射则进入刻蚀/吸附判定：离子直接减小 `grid_exist`（阈值降到 0 则变为真空）；中性粒子先吸附 Cl（`grid_count_cl`），Cl 饱和（≥3）后才能刻蚀

**5. 轮廓提取（`get_contour_points()`）**：先对 `grid_exist` 做高斯平滑（sigma=1），再用 `skimage.measure.find_contours` 或 matplotlib `contour` 提取 level=0.5 的等值线，返回 `(x, y)` 点列表。

**6. 主循环（`main()`）**：每 50 个 batch 同步一次 GPU、提取轮廓、保存 CSV、更新实时可视化（历史轮廓叠加显示，最新为白色，历史为红色）。

## 关键数据场说明

| 字段 | 类型 | 含义 |
|---|---|---|
| `grid_exist` | `f32` | 单元是否存在：0.0=真空，1.0=固体，阈值 0.5 |
| `grid_material` | `i32` | 材质：0=真空，1=Si 衬底，2=掩膜 |
| `grid_count_cl` | `i32` | 该单元吸附的 Cl 自由基数量（最多影响到 4） |
| `grid_temp` | `f32` | `smooth_grid` 的临时缓冲区 |

## 坐标约定

- 网格尺寸 `(ROWS, COLS) = (1000, 700)`，x 轴对应 ROWS（水平），y 轴对应 COLS（竖直，粒子从 y=1 向下运动）
- 绘图时使用 `np.transpose(rgb, (1,0,2))` 将 (x,y) 转为图像 (row,col) 格式
- `init_grid` 注释提醒：边界判断中 `i < ROWS` 不能写成 `i < ROWS-1`，否则最后一列不会填充
