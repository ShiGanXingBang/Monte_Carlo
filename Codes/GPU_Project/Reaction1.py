import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置 =================
CSV_FILE = "contour_0.csv"
FLIP_COORDINATES = True  # 开启坐标变换：让 Y=0 在底部，Y大在顶部
# =======================================

def get_surface_normal_cpu(grid, px, py):
    rows, cols = grid.shape
    nx, ny = 0.0, 0.0
    
    # 扫描 5x5 邻域
    for i in range(-2, 3):
        for j in range(-2, 3):
            ni, nj = px + i, py + j
            if 0 <= ni < rows and 0 <= nj < cols:
                # 核心逻辑：寻找真空重心
                # 如果 grid[ni, nj] < 0.5 (真空)，则累加方向向量
                if grid[ni, nj] < 0.5:
                    nx += float(i)
                    ny += float(j)
    
    # 归一化
    norm = np.sqrt(nx**2 + ny**2) + 1e-6
    return nx / norm, ny / norm

def main():
    print(f"Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print("Error: 找不到文件")
        return

    # --- 1. 坐标变换 (Transform) ---
    # 确定画布高度
    height_limit = int(np.ceil(df['y'].max())) + 20
    
    if FLIP_COORDINATES:
        print(">>> 执行坐标变换: 翻转 Y 轴 (Y=0 set to Bottom) ...")
        # 变换公式: y' = H - y
        # 这样原来的底部变成顶部，原来的顶部变成底部
        df['y_trans'] = height_limit - df['y']
    else:
        df['y_trans'] = df['y']

    # --- 2. 建立网格 ---
    max_x = int(np.ceil(df['x'].max())) + 10
    max_y = height_limit + 5
    grid = np.zeros((max_x, max_y))
    
    # --- 3. 网格重建 (Material Fill Logic) ---
    print("Reconstructing Grid...")
    # 取整
    df['ix'] = df['x'].round().astype(int)
    df['iy'] = df['y_trans'].round().astype(int)
    
    surface_map = df.groupby('ix')['iy'].max() if FLIP_COORDINATES else df.groupby('ix')['iy'].min()
    
    for x in range(max_x):
        if x in surface_map.index:
            y_surf = surface_map[x]
            
            if FLIP_COORDINATES:
                # 变换后：Y代表高度。表面以下(0 到 y_surf)是材料
                if y_surf > 0:
                    grid[x, 0:y_surf] = 1.0 
            else:
                # 原始模式：Y代表深度。表面以下(y_surf 到 max)是材料
                grid[x, y_surf:] = 1.0

    # --- 4. 计算法线 ---
    print("Calculating Normals...")
    nx_list, ny_list = [], []
    for _, row in df.iterrows():
        px, py = int(round(row['x'])), int(round(row['y_trans']))
        if 0 <= px < max_x and 0 <= py < max_y:
            nx, ny = get_surface_normal_cpu(grid, px, py)
            nx_list.append(nx)
            ny_list.append(ny)
        else:
            nx_list.append(0); ny_list.append(0)

    # --- 5. 绘图 (Cartesian Style) ---
    plt.figure(figsize=(12, 6))
    
    # 画背景：使用 origin='lower' 让 (0,0) 在左下角！符合物理直觉
    # grid.T 转置后，x是横轴，y是纵轴
    plt.imshow(grid.T, cmap='binary', origin='lower', alpha=0.3, interpolation='nearest')
    
    # 画箭头
    step = 20
    sub_df = df.iloc[::step]
    sub_nx = nx_list[::step]
    sub_ny = ny_list[::step]
    
    plt.quiver(sub_df['x'], sub_df['y_trans'], sub_nx, sub_ny, 
               color='red', 
               angles='xy', scale_units='xy', scale=0.15, 
               width=0.003, headwidth=4,
               label='Normal Vector')
    
    # 画轮廓线
    plt.plot(df['x'], df['y_trans'], color='blue', lw=1, alpha=0.5, label='Surface')

    # 强制等比例
    plt.gca().set_aspect('equal')
    
    plt.title(f"Coordinate Transformed View\n(Origin at Bottom-Left, Arrow points to Vacuum)")
    plt.xlabel("X")
    plt.ylabel("Y (Height)")
    plt.xlim(0, max_x)
    plt.ylim(0, max_y) # Y轴向上增加
    plt.legend()
    plt.tight_layout()
    
    out_file = "normal_transformed.png"
    plt.savefig(out_file, dpi=150)
    print(f"Done! Saved to {out_file}")
    plt.show()

if __name__ == "__main__":
    main()