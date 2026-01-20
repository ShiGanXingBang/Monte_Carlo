import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 配置 =================
CSV_FILE = "Csv\TEST2026.1.18_CD100u3\contour_26500000.csv"
# 你的GPU算法逻辑：真空重心法
# 如果 y=0 在顶部，真空在上方，那么 vacuum neighbors 的 y 都是负的相对值
# 所以 sum(ny) 是负数。
# 在图像坐标系（y向下增加）中，负数 = 向上 = 指向真空。
# 所以逻辑本身是完美的，不需要反转。
INVERT_NORMAL = False 
# ===========================================

def get_surface_normal_cpu(grid, px, py):
    rows, cols = grid.shape
    nx, ny = 0.0, 0.0
    
    # 扫描 5x5 邻域 (对应 ti.ndrange((-2, 3), (-2, 3)))
    for i in range(-2, 3):
        for j in range(-2, 3):
            ni, nj = px + i, py + j
            if 0 <= ni < rows and 0 <= nj < cols:
                # 核心：只累加真空 (grid < 0.5) 的相对位置
                if grid[ni, nj] < 0.5:
                    nx += float(i)
                    ny += float(j)
    
    if INVERT_NORMAL:
        nx, ny = -nx, -ny

    norm = np.sqrt(nx**2 + ny**2) + 1e-6
    return nx / norm, ny / norm

def main():
    print(f"Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print("Error: 找不到文件")
        return

    # --- A. 建立精确的仿真网格 ---
    # 这里的 +5 是为了留出 5x5 扫描的边界，防止越界
    max_x = int(np.ceil(df['x'].max())) + 5
    max_y = int(np.ceil(df['y'].max())) + 10
    grid = np.zeros((max_x, max_y))
    
    # --- B. 网格重建 ---
    # 假设：CSV 里的点是表面。Y 坐标比表面大的（下方）是材料。
    print("Reconstructing Grid...")
    df['ix'] = df['x'].round().astype(int) # 四舍五入取整，防止 0.5 误差
    df['iy'] = df['y'].round().astype(int)
    
    surface_map = df.groupby('ix')['iy'].min()
    
    for x in range(max_x):
        if x in surface_map.index:
            y_surf = surface_map[x]
            # 填充材料：从表面 y_surf 到底部 max_y
            if y_surf < max_y:
                grid[x, y_surf:] = 1.0 

    # --- C. 计算法线 ---
    print("Calculating Normals...")
    nx_list, ny_list = [], []
    for _, row in df.iterrows():
        px, py = int(round(row['x'])), int(round(row['y']))
        if 0 <= px < max_x and 0 <= py < max_y:
            nx, ny = get_surface_normal_cpu(grid, px, py)
            nx_list.append(nx)
            ny_list.append(ny)
        else:
            nx_list.append(0); ny_list.append(0)

    # --- D. 严谨绘图 ---
    print("Plotting... (Scale locked to Equal)")
    plt.figure(figsize=(15, 6)) # 宽长一点，适应你的数据形状
    
    # 1. 画背景网格
    # grid.T 转置后，x是横轴，y是纵轴
    # cmap: binary (0/白色=真空, 1/黑色=材料) -> 我们反一下，用 Greens
    # 让 0(真空) = 浅绿色, 1(材料) = 深绿色
    plt.imshow(grid.T, cmap='GnBu', origin='upper', alpha=0.6, interpolation='none')
    
    # 2. 画箭头
    # 关键设置：angles='xy', scale_units='xy', scale=0.5
    # 这保证了箭头方向和数据坐标系严格对应，不会被拉伸变形！
    step = 20 # 降采样，每20个画一个
    sub_df = df.iloc[::step]
    sub_nx = nx_list[::step]
    sub_ny = ny_list[::step]
    
    plt.quiver(sub_df['x'], sub_df['y'], sub_nx, sub_ny, 
               color='red', 
               angles='xy', scale_units='xy', scale=0.1, # scale越小箭头越长
               width=0.003, headwidth=4,
               label='Normal Vector')
    
    # 3. 标记表面
    plt.plot(df['x'], df['y'], color='blue', lw=1, alpha=0.3, label='Surface Contour')

    # 4. 【核心】强制锁定纵横比
    plt.gca().set_aspect('equal')
    
    plt.title("Surface Normal Verification\n(Green=Vacuum, Blue=Material, Red=Normal)")
    plt.xlabel("X")
    plt.ylabel("Y (Depth)")
    # 强制 Y 轴范围，确保上面是 0
    plt.ylim(max_y, 0)
    plt.xlim(0, max_x)
    plt.legend()
    plt.tight_layout()
    
    out_file = "normal_final_check.png"
    plt.savefig(out_file, dpi=150)
    print(f"验证图已保存至: {out_file}")
    plt.show()

if __name__ == "__main__":
    main()