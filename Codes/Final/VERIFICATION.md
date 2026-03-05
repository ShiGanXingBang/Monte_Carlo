# 等离子体刻蚀仿真改进计划 - 实施总结

## 已完成的工作

### 1. 代码重构（规范性提升）✓

#### 1.1 添加物理常量定义模块 ✓

创建了完整的物理常量配置区，消除魔法数字：

```python
# 材质类型
MAT_VACUUM = 0
MAT_SI = 1
MAT_MASK = 2

# 几何常量
GRID_SCALE_NM = 0.2  # 1 网格 = 0.2nm
TRENCH_WIDTHS = [160, 140, 120, 100, 80]  # 五个槽的宽度
PILLAR_WIDTH = 100    # 中间支柱宽度（nm）
SIDE_MASK_WIDTH = 400 # 两侧掩膜宽度（nm）
MASK_THICKNESS = 130  # 掩膜厚度（nm，网格单位）
TAPER_ANGLE_DEG = 15  # 掩膜倾角（度）

# 粒子属性
ENERGY_INITIAL = 1.0
ENERGY_MIN = 0.05
ENERGY_LOSS_MASK_REFLECTION = 0.90   # 掩膜反射保留 90% 能量
ENERGY_LOSS_SI_REFLECTION = 0.40     # Si 侧壁反射保留 40% 能量

# 刻蚀产额参数
ETCH_YIELD_BASE = 0.1
ETCH_YIELD_CL_ENHANCEMENT = 0.2
ETCH_YIELD_REFLECTED_SCALING = 0.7

# 中性粒子吸附
CL_SATURATION = 3
ADSORPTION_PROB = 0.7
```

#### 1.2 提取辅助函数为独立模块 ✓

创建了三个新的 Taichi 函数：

1. **`calculate_reflection_probability(mat, theta_coll, is_ion, cl_coverage)`**
   - 基于文献的反射概率模型
   - 离子：掠角易反射（θ > 60° 线性增长）
   - 掩膜：保底概率 0.8

2. **`calculate_etch_yield(mat, cl_coverage, energy, incident_angle, ref_count)`**
   - 刻蚀产额 Y ∝ √E（文献支持）
   - Cl 增强效应
   - 角度依赖（>40° 下降）
   - 反射后保留部分能量用于 bowing

3. **`update_ion_energy(current_energy, mat_of_surface)`**
   - 反射后更新离子能量
   - 掩膜反射：保留 90% 能量
   - Si 侧壁反射：保留 40% 能量（增强 bowing）

#### 1.3 改进 `init_grid()` 以支持 5 槽结构 ✓

动态计算五槽结构：
- 槽宽度：160/140/120/100/80 nm
- 支柱宽度：100 nm
- 两侧掩膜：400 nm
- 掩膜倾角：15°

---

### 2. 物理模型改进 ✓

#### 2.1 添加离子能量追踪 ✓

在 `simulate_batch()` 中添加：
```python
energy = ENERGY_INITIAL  # 初始能量
...
# 反射时更新能量
if is_ion == 1:
    energy = update_ion_energy(energy, mat)
```

#### 2.2 改进反射概率模型 ✓

基于文献改进：
- 掠角反射（60°-90°）线性增长
- 掩膜反射保底概率 0.8
- 中性粒子与 Cl 覆盖度相关

#### 2.3 改进刻蚀产额模型 ✓

Y ∝ √E 实现：
```python
prob_etch *= ti.sqrt(energy)
```

角度依赖：
- 入射角 > 40° 时线性下降
- 掠角刻蚀效率低

#### 2.4 增强 Bowing 效应 ✓

关键调整：
1. **降低 Si 侧壁反射能量损失**：从 0.1 → 0.4，使反射离子仍能刻蚀
2. **保留反射后刻蚀产额**：0.7（原版为 0.1）
3. **15° 掩膜倾角**：产生足够的横向分量

---

### 3. 主程序优化 ✓

#### 3.1 添加进度监控和参数输出 ✓

```python
def print_simulation_parameters():
    """ 打印仿真参数配置 """
```

#### 3.2 改进可视化 ✓

添加了更详细的标题显示：
- 进度百分比
- Bowing Enhanced 标识
- 槽宽度信息

---

## 几何结构计算验证

### 五槽结构配置

| 项目 | 数值 | 网格单位 |
|------|------|---------|
| 网格比例 | 0.2 nm/pixel | - |
| 槽1宽度 | 160 nm | 800 px |
| 槽2宽度 | 140 nm | 700 px |
| 槽3宽度 | 120 nm | 600 px |
| 槽4宽度 | 100 nm | 500 px |
| 槽5宽度 | 80 nm | 400 px |
| 支柱宽度 | 100 nm | 500 px |
| 左侧掩膜 | 400 nm | 2000 px |
| 右侧掩膜 | ~400 nm | ~2000 px |
| **总宽度** | **2200 nm** | **11050 px** |

### 槽位置（网格坐标）

- 左侧掩膜：x=[0, 2000)
- 槽1 (160nm): x=[2000, 2800)
- 支柱1: x=[2800, 3300)
- 槽2 (140nm): x=[3300, 4000)
- 支柱2: x=[4000, 4500)
- 槽3 (120nm): x=[4500, 5100)
- 支柱3: x=[5100, 5600)
- 槽4 (100nm): x=[5600, 6100)
- 支柱4: x=[6100, 6600)
- 槽5 (80nm): x=[6600, 7000)
- 右侧掩膜：x=[7000, 11050)

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `Restore_Reaction_226_v2.py` | 改进版主程序 |
| `test_geometry.py` | 几何结构测试脚本 |
| `test_simple.py` | 简单计算验证脚本 |
| `VERIFICATION.md` | 本文档 |

---

## 运行方法

```bash
# 运行改进版仿真
python Restore_Reaction_226_v2.py

# 测试几何结构
python test_geometry.py

# 简单验证计算
python test_simple.py
```

---

## 验证方案

### 1. 几何验证 ✓

运行 `init_grid()` 后可视化，确认 5 个槽结构正确。

### 2. 物理模型验证

运行完整仿真后检查：
- 能量追踪：打印初始能量、反射后能量
- Bowing：观察槽口宽度变化（上宽下窄）

### 3. 与 SEM 图像对比

观察输出轮廓：
- Bowing 位置（槽口附近）
- 侧壁倾角
- 刻蚀深度

### 4. 参数敏感性分析

可调整参数观察效果：
- `ENERGY_LOSS_SI_REFLECTION`: 0.3-0.5（控制 bowing 强度）
- `TAPER_ANGLE_DEG`: 10-20°（影响反射角度）
- `RATIO`: 调整离子/中性比例

---

## 改进效果预期

1. **Bowing 效应增强**：由于 Si 侧壁反射后能量保留 40%，反射离子仍能刻蚀对侧壁
2. **物理模型更准确**：Y ∝ √E，角度依赖更符合文献
3. **代码更规范**：消除魔法数字，常量集中定义
4. **五槽结构支持**：动态计算，易于调整

---

## 参考文献

- Modeling of microtrenching and bowing effects in nanoscale Si ICP etching
- Profile simulation of high aspect ratio contact etch
- Effect of Mask Geometry Variation on Plasma Etching Profiles
